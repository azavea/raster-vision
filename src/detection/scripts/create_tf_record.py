import csv
import logging
import os
import io
from collections import OrderedDict
import argparse

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from settings import max_num_classes, line_thickness


def create_tf_example(image_dir, image_file_name, boxes, category_index):
        use_normalized_coordinates=True, line_thickness=line_thickness)
    image_path = os.path.join(image_dir, image_file_name)
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)

    width, height = image.size
    filename = image_file_name
    encoded_image_data = encoded_jpg
    image_format = image.format.lower().encode('utf8')

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    for box in boxes:
        xmins.append(box.xmin / width)
        xmaxs.append(box.xmax / width)
        ymins.append(box.ymin / height)
        ymaxs.append(box.ymax / height)
        class_id = int(box.class_id)
        classes_text.append(
            category_index[class_id]['name'].encode('utf8'))
        classes.append(class_id)

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
                     image_dir, image_file_names):
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_file_name in enumerate(image_file_names):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(image_file_names))
        tf_example = create_tf_example(
            image_dir, image_file_name, annotations[image_file_name],
            category_index)
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


def create_tf_records(data_dir, output_dir, label_map_path):
    label_map = label_map_util.load_labelmap(label_map_path)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=max_num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    logging.info('Reading from dataset.')
    image_dir = os.path.join(data_dir, 'images')
    annotations_path = os.path.join(data_dir, 'annotations.csv')
    annotations = read_annotations(annotations_path)

    image_file_names = list(annotations.keys())
    num_examples = len(image_file_names)
    train_ratio = 0.7
    num_train = int(train_ratio * num_examples)
    train_file_names = image_file_names[:num_train]
    val_file_names = image_file_names[num_train:]
    logging.info('%d training and %d validation examples.',
                 len(train_file_names), len(val_file_names))

    train_output_path = os.path.join(output_dir, 'train.record')
    val_output_path = os.path.join(output_dir, 'val.record')

    create_tf_record(train_output_path, category_index, annotations,
                     image_dir, train_file_names)
    create_tf_record(val_output_path, category_index, annotations,
                     image_dir, val_file_names)


def parse_args():
    description = """
        Convert training chips and CSV into TFRecord format which TF needs.
    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--label-map-path')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    create_tf_records(
        args.data_dir, args.output_dir, args.label_map_path)
