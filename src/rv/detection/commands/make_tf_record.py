import csv
import io
from os.path import join, basename
from os import makedirs
import glob
import random

import PIL.Image
import tensorflow as tf
from scipy.misc import imread
import numpy as np

from object_detection.utils import (
    dataset_util, label_map_util, visualization_utils as vis_util)
from object_detection.utils.np_box_list import BoxList
from object_detection.utils.np_box_list_ops import scale

from rv.detection.commands.settings import max_num_classes, line_thickness
from rv.utils.misc import save_img

random.seed(12345)


def make_debug_plot(debug_chip_path, chip_path, norm_boxlist, category_index):
    im = imread(chip_path)
    scores = [1.0] * norm_boxlist.get().shape[0]
    vis_util.visualize_boxes_and_labels_on_image_array(
        im, norm_boxlist.get(), norm_boxlist.get_field('classes'), scores,
        category_index, use_normalized_coordinates=True,
        line_thickness=line_thickness, max_boxes_to_draw=None)
    save_img(debug_chip_path, im)


def create_tf_example(chip_set_index, chip_dir, chip_filename, boxlist,
                      category_index, debug_dir):
    # boxlist can be None if chip has no boxes in it.
    chip_path = join(chip_dir, chip_filename)
    with tf.gfile.GFile(chip_path, 'rb') as fid:
        encoded_png = fid.read()
    image = PIL.Image.open(io.BytesIO(encoded_png))

    width, height = image.size
    chip_id = '{}_{}'.format(chip_set_index, chip_filename)
    image_format = image.format.lower().encode('utf8')

    # Convert to normalized coordinates (ie. between 0 and 1).
    norm_boxlist = BoxList(np.zeros((0, 4)))
    norm_boxlist.add_field('classes', np.zeros((0,)))
    if boxlist is not None:
        norm_boxlist = scale(boxlist, 1 / height, 1 / width)
    ymins, xmins, ymaxs, xmaxs = norm_boxlist.get_coordinates()

    classes = norm_boxlist.get_field('classes')
    class_texts = [category_index[class_id]['name'].encode('utf8')
                   for class_id in classes]

    # TODO make debug chips from TFRecord.
    if debug_dir is not None:
        debug_chip_path = join(debug_dir, chip_id)
        make_debug_plot(
            debug_chip_path, chip_path, norm_boxlist, category_index)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_png),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            class_texts),
        'image/object/class/label': dataset_util.int64_list_feature(
            classes)
    }))
    return tf_example


def create_tf_examples(chip_set_index, chip_dir, chip_label_path,
                       category_index, debug_dir):
    # Get paths to chips and mapping from chips to boxes.
    chip_paths = glob.glob(join(chip_dir, '*.png'))
    chip_filenames = [basename(path) for path in chip_paths]
    filename_to_boxlist = load_chip_labels(chip_label_path)

    # Make an example for each chip.
    tf_examples = []
    for chip_filename in chip_filenames:
        tf_example = create_tf_example(
            chip_set_index, chip_dir, chip_filename,
            filename_to_boxlist.get(chip_filename), category_index,
            debug_dir)
        tf_examples.append(tf_example)

    return tf_examples

def load_chip_labels(chip_labels_path):
    filename_to_boxes = {}
    filename_to_classes = {}
    # Convert CSV file with row for each box to a map from
    # chip filename to boxlist.
    with open(chip_labels_path) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            filename = row['filename']

            boxes = filename_to_boxes.get(filename, [])
            box = [row['ymin'], row['xmin'], row['ymax'], row['xmax']]
            boxes.append(box)
            filename_to_boxes[filename] = boxes

            classes = filename_to_classes.get(filename, [])
            classes.append(row['class_id'])
            filename_to_classes[filename] = classes

    filename_to_boxlist = {}
    for filename in filename_to_boxes.keys():
        boxes = np.array(filename_to_boxes[filename], dtype=float)
        classes = np.array(filename_to_classes[filename], dtype=int)
        boxlist = BoxList(boxes)
        boxlist.add_field('classes', classes)
        filename_to_boxlist[filename] = boxlist

    return filename_to_boxlist


def write_tf_record(tf_examples, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for tf_example in tf_examples:
        writer.write(tf_example.SerializeToString())
    writer.close()


def make_tf_record(label_map_path, chip_dirs, chip_label_paths,
                   output_dir, config):
    print('Making TFRecord file...')

    # Load label map.
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Setup debug directory.
    debug_dir = None
    if config.debug is not None:
        debug_dir = join(output_dir, 'debug')
        makedirs(debug_dir, exist_ok=True)

    # For each chip set, add to the set of TFExamples.
    tf_examples = []
    for chip_set_index, (chip_dir, chip_label_path) in \
            enumerate(zip(chip_dirs, chip_label_paths)):
        print('Working on images for set #{}...'.format(chip_set_index))
        tf_examples.extend(create_tf_examples(
            chip_set_index, chip_dir, chip_label_path, category_index,
            debug_dir))

    # Write all TFExamples to TFRecord file.
    output_path = join(output_dir, 'data.record')
    write_tf_record(tf_examples, output_path)
