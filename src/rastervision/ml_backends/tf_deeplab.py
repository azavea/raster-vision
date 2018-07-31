import io
import numpy as np
import shutil
import tempfile
import tensorflow as tf
import uuid

from os.path import join
from PIL import Image
from subprocess import Popen
from tensorflow.core.example.example_pb2 import Example
from typing import (List, Tuple)

from object_detection.utils import dataset_util
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.core.ml_backend import MLBackend
from rastervision.core.scene import Scene
from rastervision.core.training_data import TrainingData
from rastervision.ml_backends.tf_object_detection_api import (
    terminate_at_exit, TRAIN, VALIDATION)
from rastervision.utils.files import make_dir
from rastervision.utils.misc import save_img


def numpy_to_png(array: np.ndarray) -> str:
    """Get a PNG string from a Numpy array.

    Args:
         array: A Numpy array of shape (w, h, 3) or (w, h), where the
               former is meant to become a three-channel image and the
               latter a one-channel image.  The dtype of the array
               should be uint8.

    Returns:
         str

    """
    im = Image.fromarray(array)
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png: str, dtype=np.uint8) -> np.ndarray:
    """Get a Numpy array from a PNG string.

    Args:
         png: A str containing a PNG-formatted image.

    Returns:
         numpy.ndarray

    """
    incoming = io.BytesIO(png)
    im = Image.open(incoming)
    return np.array(im)


def write_tf_record(tf_examples: List[Example], output_path: str) -> None:
    """Write an array of TFRecords to the given output path.

    Args:
         tf_examples: An array of TFRecords; a
              list(tensorflow.core.example.example_pb2.Example)
         output_path: The path where the records should be stored.

    Returns:
         None

    """
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())


def make_tf_examples(training_data: TrainingData,
                     class_map: ClassMap) -> List[Example]:
    """Take training data and a class map and return a list of TFRecords.

    Args:
         training_data: A rastervision.core.training_data.TrainingData
              object.
         class_map: A rastervision.core.class_map.ClassMap object.

    Returns:
         list(tensorflow.core.example.example_pb2.Example)

    """
    tf_examples = []
    print('Creating TFRecord', end='', flush=True)
    for chip, window, labels in training_data:
        tf_example = create_tf_example(chip, window, labels, class_map)
        tf_examples.append(tf_example)
        print('.', end='', flush=True)
    print()
    return tf_examples


def merge_tf_records(output_path: str, src_records: List[str]) -> None:
    """Merge mutiple TFRecord files into one.

    Args:
         output_path: Where to write the merged TFRecord file.
         src_records: A list of strings giving the location of the
              input TFRecord files.

    Returns:
         None

    """
    with tf.python_io.TFRecordWriter(output_path) as writer:
        print('Merging TFRecords', end='', flush=True)
        for src_record in src_records:
            for string_record in tf.python_io.tf_record_iterator(src_record):
                writer.write(string_record)
            print('.', end='', flush=True)
        print()


def make_debug_images(record_path: str, output_dir: str,
                      p: float = 0.25) -> None:
    """Render a random sample of the TFRecords in a given file as
    human-viewable PNG files.

    Args:
         record_path: Path to the TFRecord file.
         output_dir: Destination directory for the generated PNG files.
         p: The probability of rendering a particular record.

    Return:
         None

    """
    make_dir(output_dir, check_empty=True)

    print('Generating debug chips', end='', flush=True)
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        example = tf.train.Example.FromString(example)
        im, labels = parse_tf_example(example)
        output_path = join(output_dir, '{}.png'.format(ind))
        inv_labels = (labels == 0)
        im[:, :, 0] = im[:, :, 0] * inv_labels
        im[:, :, 1] = im[:, :, 1] * inv_labels
        im[:, :, 2] = im[:, :, 2] * inv_labels
        if np.random.rand() <= p:
            save_img(im, output_path)
        print('.', end='', flush=True)
    print()


def parse_tf_example(example: Example) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a TensorFlow Example into an image array and a label array.

    Args:
         example: A TensorFlow Example object.

    Return:
         tuple(np.ndarray, np.ndarray)
    """
    ie = 'image/encoded'
    isce = 'image/segmentation/class/encoded'
    image_encoded = \
        example.features.feature[ie].bytes_list.value[0]
    image_segmentation_class_encoded = \
        example.features.feature[isce].bytes_list.value[0]
    im = png_to_numpy(image_encoded)
    labels = png_to_numpy(image_segmentation_class_encoded)
    return im, labels


def create_tf_example(image: np.ndarray,
                      window: Box,
                      labels: np.ndarray,
                      class_map: ClassMap,
                      chip_id: str = '') -> Example:
    """Create a TensorFlow Example from an image, the labels, &c.

    Args:
         image: An nd.array containing the image data.
         window: A Box object containing the bounding box for this example.
         labels: An nd.array containing the label data.
         class_map: A ClassMap object containing mappings between
              numerial and textual labels.
         chip_id: The chip id as a string.

    Returns:
         A Deeplab-compatible TensorFlow Example object containing the
              given data.

    """
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


class TFDeeplab(MLBackend):
    """MLBackend-derived type that implements the TensorFlow DeepLab
    backend.

    """

    def __init__(self):
        """Constructor"""
        # persist scene training packages for when output_uri is remote
        self.scene_training_packages = []

    def process_scene_data(self, scene: Scene, data: TrainingData,
                           class_map: ClassMap, options) -> str:
        """Process the given scene and data into a TFRecord file specifically
        associated with that file.

        Args:
             scene: The scene data (labels stores, the raster sources,
                  and so on).
             data: The training data.
             class_map: The mapping from numerical labels to textual
                  labels.
             options: The options given to `make_training_chips` in
                  the config file.

        Returns:
            The path to the generated file.

        """
        base_uri = options.output_uri
        make_dir(base_uri)

        tf_examples = make_tf_examples(data, class_map)
        split = '{}-{}'.format(scene.id, uuid.uuid4())
        record_path = join(base_uri, '{}.record'.format(split))
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results: List[str],
                                 validation_results: List[str],
                                 class_map: ClassMap, options) -> None:
        """Merge TFRecord files from individual scenes into two at-large files
        (one for training data and one for validation data).

        Args:

             training_results: A list of paths to TFRecords containing
                  training data.
             valiation_results: A list of paths to TFRecords
                  containing validation data.
             class_map: A mapping from numerical classes to their
                  textual names.
             options: The options given to `make_training_chips` in
                  the config file.

        Returns:
             None
        """
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
        import pdb
        pdb.set_trace()
        soptions = options.segmentation_options

        train_logdir = options.output_uri
        dataset_dir = options.training_data_uri
        train_py = soptions.train_py
        tf_initial_checkpoints = soptions.tf_initial_checkpoint

        args = ['python', train_py]
        args.append('--train_logdir={}'.format(train_logdir))
        args.append(
            '--tf_initial_checkpoint={}'.format(tf_initial_checkpoints))
        args.append('--dataset_dir={}'.format(dataset_dir))
        args.append('--training_number_of_steps={}'.format(
            soptions.training_number_of_steps))
        if len(soptions.train_split) > 0:
            args.append('--train_split="{}"'.format(soptions.train_split))
        if len(soptions.model_variant) > 0:
            args.append('--model_variant="{}"'.format(soptions.model_variant))
        for rate in soptions.atrous_rates:
            args.append('--atrous_rates={}'.format(rate))
        args.append('--output_stride={}'.format(soptions.output_stride))
        args.append('--decoder_output_stride={}'.format(
            soptions.decoder_output_stride))
        for size in soptions.train_crop_size:
            args.append('--train_crop_size={}'.format(size))
        args.append('--train_batch_size={}'.format(soptions.train_batch_size))
        if len(soptions.dataset):
            args.append('--dataset="{}"'.format(soptions.dataset))

        train_process = Popen(args)
        terminate_at_exit(train_process)

        # XXX tensorboard

        train_process.wait()

    def predict(self, chip, options):
        return 1
