import csv
import logging
import os
import io
from collections import OrderedDict

import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw pet dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/ships_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS


def create_tf_example(image_dir, image_file_name, boxes, label_map_dict):
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
        xmins.append(box.xmin)
        xmaxs.append(box.xmax)
        ymins.append(box.ymin)
        ymaxs.append(box.ymax)
        classes_text.append(box.class_name.encode('utf8'))
        classes.append(label_map_dict[box.class_name])

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


def create_tf_record(output_path, label_map_dict, annotations,
                     image_dir, image_file_names):
    writer = tf.python_io.TFRecordWriter(output_path)
    for idx, image_file_name in enumerate(image_file_names):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(image_file_names))
        tf_example = create_tf_example(
            image_dir, image_file_name, annotations[image_file_name],
            label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()


class Box():
    def __init__(self, xmin, xmax, ymin, ymax, class_name):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.class_name = class_name


def read_annotations(annotations_path):
    annotations = OrderedDict()
    with open(annotations_path) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row in reader:
            filename = row['filename']
            boxes = annotations.get(filename, [])
            boxes.append(Box(
                int(row['xmin']), int(row['xmax']), int(row['ymin']),
                int(row['ymax']), row['class_name']))
            annotations[filename] = boxes
    return annotations


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    logging.info('Reading from Ships dataset.')
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

    train_output_path = os.path.join(FLAGS.output_dir, 'ship_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'ship_val.record')

    create_tf_record(train_output_path, label_map_dict, annotations,
                     image_dir, train_file_names)
    create_tf_record(val_output_path, label_map_dict, annotations,
                     image_dir, val_file_names)


if __name__ == '__main__':
    tf.app.run()
