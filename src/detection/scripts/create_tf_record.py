import csv
import logging
import os
import io
from collections import OrderedDict
import argparse
from os.path import join, basename
from os import makedirs
import glob
import random

import PIL.Image
import tensorflow as tf
from scipy.misc import imread, imsave
import numpy as np

from object_detection.utils import (
    dataset_util, label_map_util, visualization_utils as vis_util)

from settings import max_num_classes, line_thickness

random.seed(12345)


def make_debug_plot(debug_image_path, image_path, xmins, xmaxs, ymins, ymaxs,
                    classes, category_index):
    im = imread(image_path)
    boxes = np.array(list(zip(ymins, xmins, ymaxs, xmaxs)))
    scores = np.array([1.0] * len(classes))

    vis_util.visualize_boxes_and_labels_on_image_array(
        im, boxes, classes, scores, category_index,
        use_normalized_coordinates=True, line_thickness=line_thickness)
    imsave(debug_image_path, im)


def create_tf_example(image_dir, image_file_name, boxes, category_index,
                      debug_dir):
    image_path = os.path.join(image_dir, image_file_name)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_png = fid.read()
    encoded_png_io = io.BytesIO(encoded_png)
    image = PIL.Image.open(encoded_png_io)

    width, height = image.size
    filename = image_file_name
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
        debug_image_path = join(debug_dir, image_file_name)
        make_debug_plot(
            debug_image_path, image_path, xmins, xmaxs, ymins, ymaxs, classes,
            category_index)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
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
                     image_dir, image_file_names, debug_dir):
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_file_name in enumerate(image_file_names):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(image_file_names))
        tf_example = create_tf_example(
            image_dir, image_file_name, annotations.get(image_file_name),
            category_index, debug_dir)
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


def create_tf_records(data_dir, debug):
    label_map_path = join(data_dir, 'label_map.pbtxt')
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    logging.info('Reading from dataset.')
    image_dir = os.path.join(data_dir, 'images')
    image_paths = glob.glob(join(image_dir, '*.png'))
    image_file_names = [basename(path) for path in image_paths]
    random.shuffle(image_file_names)
    annotations_path = os.path.join(data_dir, 'annotations.csv')
    annotations = read_annotations(annotations_path)

    num_examples = len(image_file_names)
    train_ratio = 0.7
    num_train = int(train_ratio * num_examples)
    train_file_names = image_file_names[:num_train]
    val_file_names = image_file_names[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_file_names), len(val_file_names))

    train_output_path = os.path.join(data_dir, 'train.record')
    val_output_path = os.path.join(data_dir, 'val.record')

    debug_dir = None
    if debug is not None:
        debug_dir = join(data_dir, 'debug')
        makedirs(debug_dir, exist_ok=True)

    create_tf_record(train_output_path, category_index, annotations,
                     image_dir, train_file_names, debug_dir)
    create_tf_record(val_output_path, category_index, annotations,
                     image_dir, val_file_names, debug_dir)


def parse_args():
    description = """
        Convert training chips and CSV into TFRecord format which TF needs.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data-dir')
    parser.add_argument('--debug', dest='debug', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    create_tf_records(args.data_dir, args.debug)
