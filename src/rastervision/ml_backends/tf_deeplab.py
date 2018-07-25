import atexit
import io
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import uuid

from os.path import join
from PIL import Image
from subprocess import Popen

from object_detection.utils import dataset_util
from rastervision.core.ml_backend import MLBackend
from rastervision.utils.files import make_dir
from rastervision.utils.misc import save_img

TRAIN = 'train'
VALIDATION = 'validation'


def numpy_to_png(array):
    im = Image.fromarray(array)
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png, dtype=np.uint8):
    incoming = io.BytesIO(png)
    im = Image.open(incoming)
    return np.array(im)


def write_tf_record(tf_examples, output_path):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())


def make_tf_examples(training_data, class_map):
    tf_examples = []
    print('Creating TFRecord', end='', flush=True)
    for chip, window, labels in training_data:
        tf_example = create_tf_example(chip, window, labels, class_map)
        tf_examples.append(tf_example)
        print('.', end='', flush=True)
    print()
    return tf_examples


def merge_tf_records(output_path, src_records):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        print('Merging TFRecords', end='', flush=True)
        for src_record in src_records:
            for string_record in tf.python_io.tf_record_iterator(src_record):
                writer.write(string_record)
            print('.', end='', flush=True)
        print()


def make_debug_images(record_path, output_dir):
    make_dir(output_dir, check_empty=True)

    print('Generating debug chips', end='', flush=True)
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        example = tf.train.Example.FromString(example)
        im, labels = parse_tfexample(example)
        output_path = join(output_dir, '{}.png'.format(ind))
        inv_labels = (labels == 0)
        im[:, :, 0] = im[:, :, 0] * inv_labels  # XXX
        im[:, :, 1] = im[:, :, 1] * inv_labels  # XXX
        im[:, :, 2] = im[:, :, 2] * inv_labels  # XXX
        save_img(im, output_path)
        print('.', end='', flush=True)
    print()


def parse_tfexample(example):
    image_encoded = example.features.feature['image/encoded'].bytes_list.value[
        0]
    image_segmentation_class_encoded = example.features.feature[
        'image/segmentation/class/encoded'].bytes_list.value[0]
    im = png_to_numpy(image_encoded)
    labels = png_to_numpy(image_segmentation_class_encoded)
    return im, labels


def create_tf_example(image, window, labels, class_map, chip_id=''):
    class_keys = set(class_map.get_keys())

    def fn(n):
        return (n if n in class_keys else 0)

    filtered_labels = np.array(np.vectorize(fn)(labels), dtype=np.uint8)

    image_encoded = numpy_to_png(image)
    image_filename = chip_id.encode('utf8')
    image_format = 'png'.encode('utf8')
    image_height, image_width, image_channels = image.shape
    image_segmentation_class_encoded = numpy_to_png(filtered_labels)
    image_segmentation_class_format = 'png'.encode('utf8')

    features = tf.train.Features(
        feature={
            'image/encoded':
            dataset_util.bytes_feature(image_encoded),
            'image/filename':
            dataset_util.bytes_feature(image_filename),
            'image/format':
            dataset_util.bytes_feature(image_format),
            'image/height':
            dataset_util.int64_feature(image_height),
            'image/width':
            dataset_util.int64_feature(image_width),
            'image/channels':
            dataset_util.int64_feature(image_channels),
            'image/segmentation/class/encoded':
            dataset_util.bytes_feature(image_segmentation_class_encoded),
            'image/segmentation/class/format':
            dataset_util.bytes_feature(image_segmentation_class_format),
        })

    return tf.train.Example(features=features)


def terminate_at_exit(process):
    def terminate():
        print('Terminating {}...'.format(process.pid))
        process.terminate()

    atexit.register(terminate)


class TFDeeplab(MLBackend):
    def __init__(self):
        # persist scene training packages for when output_uri is remote
        self.scene_training_packages = []

    def process_scene_data(self, scene, data, class_map, options):
        base_uri = options.output_uri
        make_dir(base_uri)

        tf_examples = make_tf_examples(data, class_map)
        split = '{}-{}'.format(scene.id, uuid.uuid4())
        record_path = join(base_uri, '{}.record'.format(split))
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results, validation_results,
                                 class_map, options):
        base_uri = options.output_uri

        training_record_path = join(base_uri, '{}-0.record'.format(TRAIN))
        validation_record_path = join(base_uri,
                                      '{}-0.record'.format(VALIDATION))
        merge_tf_records(training_record_path, training_results)
        merge_tf_records(validation_record_path, validation_results)

        if options.debug:
            training_zip_path = join(base_uri, '{}'.format(TRAIN))
            validation_zip_path = join(base_uri, '{}'.format(VALIDATION))
            with tempfile.TemporaryDirectory() as debug_dir:
                make_debug_images(training_record_path, debug_dir)
                shutil.make_archive(training_zip_path, 'zip', debug_dir)
            with tempfile.TemporaryDirectory() as debug_dir:
                make_debug_images(validation_record_path, debug_dir)
                shutil.make_archive(validation_zip_path, 'zip', debug_dir)

    def train(self, class_map, options):
        train_logdir = options.output_uri
        dataset_dir = options.training_data_uri
        train_py = options.segmentation_options.train_py
        tf_initial_checkpoints = \
            options.segmentation_options.tf_initial_checkpoint

        args = ['python', train_py]
        args.append('--train_logdir={}'.format(train_logdir))
        args.append(
            '--tf_initial_checkpoint={}'.format(tf_initial_checkpoints))
        args.append('--dataset_dir={}'.format(dataset_dir))

        train_process = Popen(args)
        terminate_at_exit(train_process)

        # XXX tensorboard

        train_process.wait()

    def predict(self, chip, options):
        return 1
