import csv
import logging
import io
from collections import OrderedDict
from os.path import join, basename
from os import makedirs
import glob
import random

import click
import PIL.Image
import tensorflow as tf
from scipy.misc import imread, imsave
import numpy as np

from object_detection.utils import (
    dataset_util, label_map_util, visualization_utils as vis_util)

from rv.commands.settings import max_num_classes, line_thickness

random.seed(12345)


def make_debug_plot(debug_chip_path, chip_path, xmins, xmaxs, ymins, ymaxs,
                    classes, category_index):
    im = imread(chip_path)
    boxes = np.array(list(zip(ymins, xmins, ymaxs, xmaxs)))
    scores = np.array([1.0] * len(classes))

    vis_util.visualize_boxes_and_labels_on_image_array(
        im, boxes, classes, scores, category_index,
        use_normalized_coordinates=True, line_thickness=line_thickness)
    imsave(debug_chip_path, im)


def create_tf_example(chip_set_index, chip_dir, chip_file_name, boxes,
                      category_index, debug_dir):
    chip_path = join(chip_dir, chip_file_name)
    with tf.gfile.GFile(chip_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = PIL.Image.open(encoded_png_io)

    width, height = image.size
    chip_id = '{}_{}'.format(chip_set_index, chip_file_name)
    encoded_image_data = encoded_png
    image_format = image.format.lower().encode('utf8')

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    if boxes is not None:
        for box in boxes:
            xmins.append(box.xmin / width)
            xmaxs.append(box.xmax / width)
            ymins.append(box.ymin / height)
            ymaxs.append(box.ymax / height)
            class_id = int(box.class_id)
            classes_text.append(
                category_index[class_id]['name'].encode('utf8'))
            classes.append(class_id)

    if debug_dir is not None:
        debug_chip_path = join(debug_dir, chip_id)
        make_debug_plot(
            debug_chip_path, chip_path, xmins, xmaxs, ymins, ymaxs, classes,
            category_index)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_path, category_index, annotations,
                     chip_set_index, chip_dir, chip_file_names, debug_dir):
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, chip_file_name in enumerate(chip_file_names):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(chip_file_names))
        tf_example = create_tf_example(
            chip_set_index, chip_dir, chip_file_name,
            annotations.get(chip_file_name), category_index, debug_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()


class Box():
    def __init__(self, xmin, xmax, ymin, ymax, class_id):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.class_id = class_id


def read_annotations(annotations_path):
    annotations = OrderedDict()
    with open(annotations_path) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            filename = row['filename']
            boxes = annotations.get(filename, [])
            boxes.append(Box(
                int(row['xmin']), int(row['xmax']), int(row['ymin']),
                int(row['ymax']), row['class_id']))
            annotations[filename] = boxes
    return annotations


@click.command()
@click.argument('label_map_path')
@click.argument('output_dir')
@click.argument('chip_label_paths', nargs=-1)
@click.option('--debug', is_flag=True,
              help='Generate debug plots that contain bounding boxes')
def make_tf_record(label_map_path, output_dir, chip_label_paths, debug):
    """
        Convert training chips and CSV into TFRecord format which TF needs.

        Args:
            chip_label_paths: alternating list of chip directory and
            path to labels. eg. /tmp/chips1 /tmp/chips1.csv /tmp/chips2
            /tmp/chips2.csv
    """
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    logging.info('Reading from dataset.')

    if len(chip_label_paths) % 2 != 0:
        raise Exception('chip_label_paths must have even number of elements')

    for chip_set_index in range(int(len(chip_label_paths) / 2)):
        chip_dir = chip_label_paths[chip_set_index * 2]
        chip_label_path = chip_label_paths[chip_set_index * 2 + 1]

        chip_paths = glob.glob(join(chip_dir, '*.png'))
        chip_file_names = [basename(path) for path in chip_paths]
        annotations = read_annotations(chip_label_path)

        random.shuffle(chip_file_names)
        num_examples = len(chip_file_names)
        train_ratio = 0.7
        num_train = int(train_ratio * num_examples)
        train_file_names = chip_file_names[:num_train]
        val_file_names = chip_file_names[num_train:]

        train_output_path = join(output_dir, 'train.record')
        val_output_path = join(output_dir, 'val.record')

        debug_dir = None
        if debug is not None:
            debug_dir = join(output_dir, 'debug')
            makedirs(debug_dir, exist_ok=True)

        create_tf_record(train_output_path, category_index, annotations,
                         chip_set_index, chip_dir, train_file_names, debug_dir)
        create_tf_record(val_output_path, category_index, annotations,
                         chip_set_index, chip_dir, val_file_names, debug_dir)


if __name__ == '__main__':
    make_tf_record()
