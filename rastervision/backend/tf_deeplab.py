import os
import glob
import shutil
import tarfile
import uuid
from typing import (Dict, List, Tuple)
from os.path import join
from subprocess import Popen
import logging

import numpy as np
from google.protobuf import (json_format)

import rastervision as rv
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.backend import Backend
from rastervision.data.scene import Scene
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.core.training_data import TrainingData
from rastervision.backend.tf_object_detection import (
    write_tf_record, terminate_at_exit, TRAIN, VALIDATION)
from rastervision.protos.deeplab.train_pb2 import (TrainingParameters as
                                                   TrainingParametersMsg)
from rastervision.utils.files import (download_if_needed, get_local_path,
                                      make_dir, start_sync, upload_or_copy,
                                      sync_to_dir, sync_from_dir)
from rastervision.utils.misc import (numpy_to_png, png_to_numpy, save_img)
from rastervision.data.label_source.utils import color_to_integer
from rastervision.rv_config import RVConfig

FROZEN_INFERENCE_GRAPH = 'model'
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'

log = logging.getLogger(__name__)


def make_tf_examples(training_data: TrainingData, class_map: ClassMap) -> List:
    """Take training data and a class map and return a list of TFRecords.

    Args:
         training_data: A rastervision.core.training_data.TrainingData
              object.
         class_map: A rastervision.core.class_map.ClassMap object.

    Returns:
         list(tensorflow.core.example.example_pb2.Example)

    """
    tf_examples = []
    log.info('Creating TFRecord')
    for chip, window, labels in training_data:
        tf_example = create_tf_example(chip, window, labels.to_array(),
                                       class_map)
        tf_examples.append(tf_example)
    return tf_examples


def merge_tf_records(output_path: str, src_records: List[str]) -> None:
    """Merge multiple TFRecord files into one.

    Args:
         output_path: Where to write the merged TFRecord file.
         src_records: A list of strings giving the location of the
              input TFRecord files.

    Returns:
         None

    """
    import tensorflow as tf

    records = 0
    with tf.python_io.TFRecordWriter(output_path) as writer:
        log.info('Merging TFRecords')
        for src_record in src_records:
            for string_record in tf.python_io.tf_record_iterator(src_record):
                writer.write(string_record)
                records = records + 1
        log.info('{} records'.format(records))


def make_debug_images(record_path: str, output_dir: str, class_map: ClassMap,
                      p: float) -> None:
    """Render a random sample of the TFRecords in a given file as
    human-viewable PNG files.

    Args:
         record_path: Path to the TFRecord file.
         output_dir: Destination directory for the generated PNG files.
         p: The probability of rendering a particular record.

    Returns:
         None

    """
    import tensorflow as tf
    make_dir(output_dir)

    ids = class_map.get_keys()
    color_strs = list(map(lambda c: c.color, class_map.get_items()))
    color_ints = list(map(lambda c: color_to_integer(c), color_strs))
    correspondence = dict(zip(ids, color_ints))

    def _label_fn(v: int) -> int:
        if v in correspondence:
            return correspondence.get(v)
        else:
            return 0

    label_fn = np.vectorize(_label_fn, otypes=[np.uint64])

    def _image_fn(pixel: int) -> int:
        if (pixel & 0x00ffffff):
            r = ((pixel >> 41 & 0x7f) + (pixel >> 17 & 0x7f)) << 16
            g = ((pixel >> 33 & 0x7f) + (pixel >> 9 & 0x7f)) << 8
            b = ((pixel >> 25 & 0x7f) + (pixel >> 1 & 0x7f)) << 0
            return r + g + b
        else:
            return pixel >> 24

    image_fn = np.vectorize(_image_fn, otypes=[np.uint64])

    log.info('Generating debug chips')
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        if np.random.rand() <= p:
            example = tf.train.Example.FromString(example)
            im_unpacked, labels = parse_tf_example(example)

            im_r = np.array(im_unpacked[:, :, 0], dtype=np.uint64) * 1 << 40
            im_g = np.array(im_unpacked[:, :, 1], dtype=np.uint64) * 1 << 32
            im_b = np.array(im_unpacked[:, :, 2], dtype=np.uint64) * 1 << 24
            im_packed = im_r + im_g + im_b

            labels_packed = label_fn(np.array(labels))
            im_labels_packed = im_packed + labels_packed
            im_packed = image_fn(im_labels_packed)

            im_unpacked[:, :, 0] = np.bitwise_and(
                im_packed >> 16, 0xff, dtype=np.uint8)
            im_unpacked[:, :, 1] = np.bitwise_and(
                im_packed >> 8, 0xff, dtype=np.uint8)
            im_unpacked[:, :, 2] = np.bitwise_and(
                im_packed >> 0, 0xff, dtype=np.uint8)

            output_path = join(output_dir, '{}.png'.format(ind))
            save_img(im_unpacked, output_path)


def parse_tf_example(example) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a TensorFlow Example into an image array and a label array.

    Args:
         example: A TensorFlow Example object.

    Returns:
         A np.ndarray × np.ndarray pair.

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
                      chip_id: str = ''):
    """Create a TensorFlow from an image, the labels, &c.

    Args:
         image: An np.ndarray containing the image data.
         window: A Box object containing the bounding box for this example.
         labels: An nd.array containing the label data.
         class_map: A ClassMap object containing mappings between
              numerical and textual labels.
         chip_id: The chip id as a string.

    Returns:
         A DeepLab-compatible TensorFlow Example object containing the
         given data.

    """
    import tensorflow as tf
    from object_detection.utils import dataset_util

    class_keys = set(class_map.get_keys())

    def _clean(n):
        return (n if n in class_keys else 0x00)

    clean = np.vectorize(_clean, otypes=[np.uint8])

    image_encoded = numpy_to_png(image)
    image_filename = chip_id.encode('utf8')
    image_format = 'png'.encode('utf8')
    image_height, image_width, image_channels = image.shape
    image_segmentation_class_encoded = numpy_to_png(clean(labels))
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


def get_record_uri(base_uri: str, split: str) -> str:
    """Given a base URI and a split, return a filename to use.

    Args:
         base_uri: The directory under-which the returned record uri
              will reside.
         split: The split ("train", "validate", et cetera).

    Returns:
         A uri, under the base_uri, that can be used to store a record
         file.

    """
    return join(base_uri, '{}-0.record'.format(split))


def get_latest_checkpoint(train_logdir_local: str) -> str:
    """Return the most recently generated checkpoint.

    Args:
         train_logir_local: The directory in-which to look for the
              latest checkpoint.

    Returns:
         Returns the (local) URI to the latest checkpoint.

    """
    ckpts = glob.glob(join(train_logdir_local, 'model.ckpt-*.meta'))
    times = map(os.path.getmtime, ckpts)
    latest = sorted(zip(times, ckpts))[-1][1]
    return latest[:len(latest) - len('.meta')]


def get_training_args(train_py: str, train_logdir_local: str, tfic_ckpt: str,
                      dataset_dir_local: str, num_classes: int,
                      tfdl_config) -> Tuple[List[str], Dict[str, str]]:
    """Generate the array of arguments needed to run the training script.

    Args:
         train_py: The URI of the training script.
         train_logdir_local: The directory in-which checkpoints will
              be placed.
         tfic_ckpt: URI of the .ckpt "file" from the initial
              checkpoint tarball.
         dataset_dir_local: The directory in which the records are
              found.
         num_classes: The number of classes.
         tfdl_config: google.protobuf.Struct with fields from
            rv.protos.deeplab.train.proto containing TF Deeplab training configuration

    Returns:
         A tuple of two things: (1) a list of arguments suitable for
         starting the training script and (2) an environment in-which
         to start the training script.

    """
    fields = [
        'fine_tune_batch_norm',
        'initialize_last_layer',
        'last_layers_contain_logits_only',
        'save_summaries_images',
        'upsample_logits',
        'base_learning_rate',
        'last_layer_gradient_multiplier',
        'learning_power',
        'learning_rate_decay_factor',
        'max_scale_factor',
        'min_scale_factor',
        'momentum',
        'scale_factor_step_size',
        'slow_start_learning_rate',
        'weight_decay',
        'decoder_output_stride',
        'learning_rate_decay_step',
        'output_stride',
        'save_interval_secs',
        'save_summaries_secs',
        'slow_start_step',
        'train_batch_size',
        'training_number_of_steps',
        'dataset',
        'learning_policy',
        'model_variant',
        'train_split',
    ]

    multi_fields = [
        'atrous_rates',
        'train_crop_size',
    ]

    env_fields = [
        'dl_custom_train',
        'dl_custom_validation',
    ]

    args = ['python', train_py]

    args.append('--train_logdir={}'.format(train_logdir_local))
    args.append('--tf_initial_checkpoint={}'.format(tfic_ckpt))
    args.append('--dataset_dir={}'.format(dataset_dir_local))

    for field in multi_fields:
        for item in tfdl_config.__getattribute__(field):
            args.append('--{}={}'.format(field, item))

    for field in fields:
        field_value = tfdl_config.__getattribute__(field)
        if (not type(field_value) is str) or (not len(field_value) == 0):
            args.append('--{}={}'.format(field, field_value))

    env = os.environ.copy()
    for field in env_fields:
        field_value = tfdl_config.__getattribute__(field)
        log.info('{}={}'.format(field.upper(), field_value))
        env[field.upper()] = str(field_value)
    log.info('DL_CUSTOM_CLASSES={}'.format(num_classes))
    env['DL_CUSTOM_CLASSES'] = str(num_classes)

    return (args, env)


def get_export_args(export_py: str, train_logdir_local: str, num_classes: int,
                    tfdl_config) -> List[str]:
    """Generate the array of arguments needed to run the export script.

    Args:
         export_py: The URI of the export script.
         train_logdir_local: The directory in-which checkpoints will
              be placed.
         num_classes: The number of classes.
         tfdl_config: google.protobuf.Struct with fields from
            rv.protos.deeplab.train.proto containing TF Deeplab training configuration

    Returns:
         A list of arguments suitable for starting the training
         script.
    """

    fields = [
        'decoder_output_stride',
        'output_stride',
        'model_variant',
    ]

    args = ['python', export_py]

    args.append('--checkpoint_path={}'.format(
        get_latest_checkpoint(train_logdir_local)))
    args.append('--export_path={}'.format(
        join(train_logdir_local, FROZEN_INFERENCE_GRAPH)))
    args.append('--num_classes={}'.format(num_classes))

    for field in fields:
        field_value = tfdl_config.__getattribute__(field)
        args.append('--{}={}'.format(field, field_value))

    for item in tfdl_config.__getattribute__('atrous_rates'):
        args.append('--{}={}'.format('atrous_rates', item))

    for item in tfdl_config.__getattribute__('train_crop_size'):
        args.append('--{}={}'.format('crop_size', item))

    return args


class TFDeeplab(Backend):
    """Backend-derived type that implements the TensorFlow DeepLab
    backend.

    """

    def __init__(self, backend_config, task_config):
        """Constructor.

        Args:
            backend_config: rv.backend.TFDeeplabConfig
            task_config: rv.task.SemanticSegmentationConfig
        """
        self.sess = None
        self.backend_config = backend_config
        self.task_config = task_config
        self.class_map = task_config.class_map

    def process_scene_data(self, scene: Scene, data: TrainingData,
                           tmp_dir: str) -> str:
        """Process the given scene and data into a TFRecord file specifically
        associated with that file.

        Args:
             scene: The scene data (labels stores, the raster sources,
                  and so on).
             data: The training data.
             tmp_dir: (str) temporary directory to use
        Returns:
            The local path to the generated file.
        """
        # Currently TF Deeplab can only handle uint8
        if scene.raster_source.get_dtype() != np.uint8:
            raise Exception('Cannot use {} backend for imagery that does '
                            'not have data type uint8. '
                            'Use the StatsAnalyzer and StatsTransformer '
                            'to turn the raster data into uint8 data'.format(
                                rv.TF_DEEPLAB))

        tf_examples = make_tf_examples(data, self.class_map)

        base_uri = self.backend_config.training_data_uri
        split = '{}-{}'.format(scene.id, uuid.uuid4())
        record_path = join(base_uri, '{}.record'.format(split))
        record_path = get_local_path(record_path, tmp_dir)

        make_dir(record_path, use_dirname=True)
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results: List[str],
                                 validation_results: List[str],
                                 tmp_dir: str) -> None:
        """Merge TFRecord files from individual scenes into two at-large files
        (one for training data and one for validation data).

        Args:
             training_results: A list of paths to TFRecords containing
                  training data.
             validation_results: A list of paths to TFRecords
                  containing validation data.
             tmp_dir: (str) temporary directory to use
        Returns:
             None

        """
        base_uri = self.backend_config.training_data_uri
        training_record_path = get_record_uri(base_uri, TRAIN)
        training_record_path_local = get_local_path(training_record_path,
                                                    tmp_dir)
        validation_record_path = get_record_uri(base_uri, VALIDATION)
        validation_record_path_local = get_local_path(validation_record_path,
                                                      tmp_dir)

        make_dir(training_record_path_local, use_dirname=True)
        make_dir(validation_record_path_local, use_dirname=True)  # sic
        merge_tf_records(training_record_path_local, training_results)
        merge_tf_records(validation_record_path_local, validation_results)
        upload_or_copy(training_record_path_local, training_record_path)
        upload_or_copy(validation_record_path_local, validation_record_path)

        if self.backend_config.debug:
            training_zip_path = join(base_uri, '{}'.format(TRAIN))
            training_zip_path_local = get_local_path(training_zip_path,
                                                     tmp_dir)
            validation_zip_path = join(base_uri, '{}'.format(VALIDATION))
            validation_zip_path_local = get_local_path(validation_zip_path,
                                                       tmp_dir)

            training_debug_dir = join(tmp_dir, 'training-debug')
            make_debug_images(
                training_record_path_local, training_debug_dir, self.class_map,
                self.task_config.chip_options.debug_chip_probability)
            shutil.make_archive(training_zip_path_local, 'zip',
                                training_debug_dir)

            validation_debug_dir = join(tmp_dir, 'validation-debug')
            make_debug_images(
                validation_record_path_local, validation_debug_dir,
                self.class_map,
                self.task_config.chip_options.debug_chip_probability)
            shutil.make_archive(validation_zip_path_local, 'zip',
                                validation_debug_dir)

            upload_or_copy('{}.zip'.format(training_zip_path_local),
                           '{}.zip'.format(training_zip_path))
            upload_or_copy('{}.zip'.format(validation_zip_path_local),
                           '{}.zip'.format(validation_zip_path))

    def train(self, tmp_dir: str) -> None:
        """Train a DeepLab model the task and backend config.

        Args:
            tmp_dir: (str) temporary directory to use

        Returns:
             None
        """
        train_py = self.backend_config.script_locations.train_py
        export_py = self.backend_config.script_locations.export_py

        # Restart support
        train_restart_dir = self.backend_config.train_options.train_restart_dir
        if type(train_restart_dir) is str and len(train_restart_dir) > 0:
            tmp_dir = train_restart_dir

        # Setup local input and output directories
        log.info('Setting up local input and output directories')
        train_logdir = self.backend_config.training_output_uri
        train_logdir_local = get_local_path(train_logdir, tmp_dir)
        dataset_dir = self.backend_config.training_data_uri
        dataset_dir_local = get_local_path(dataset_dir, tmp_dir)
        make_dir(tmp_dir)
        make_dir(train_logdir_local)
        make_dir(dataset_dir_local)

        # Download training data
        log.info('Downloading training data')
        download_if_needed(get_record_uri(dataset_dir, TRAIN), tmp_dir)

        # Download and untar initial checkpoint.
        log.info('Downloading and untarring initial checkpoint')
        tf_initial_checkpoints_uri = self.backend_config.pretrained_model_uri
        download_if_needed(tf_initial_checkpoints_uri, tmp_dir)
        tfic_tarball = get_local_path(tf_initial_checkpoints_uri, tmp_dir)
        tfic_dir = os.path.dirname(tfic_tarball)
        with tarfile.open(tfic_tarball, 'r:gz') as tar:
            tar.extractall(tfic_dir)
        tfic_ckpt = glob.glob('{}/*/*.index'.format(tfic_dir))[0]
        tfic_ckpt = tfic_ckpt[0:-len('.index')]

        # Get output from potential previous run so we can resume training.
        if type(train_restart_dir) is str and len(
                train_restart_dir
        ) > 0 and not self.backend_config.train_options.replace_model:
            sync_from_dir(train_logdir, train_logdir_local)
        else:
            if self.backend_config.train_options.replace_model:
                if os.path.exists(train_logdir_local):
                    shutil.rmtree(train_logdir_local)
                make_dir(train_logdir_local)

        # Periodically synchronize with remote
        sync = start_sync(
            train_logdir_local,
            train_logdir,
            sync_interval=self.backend_config.train_options.sync_interval)

        with sync:
            # Setup TFDL config
            tfdl_config = json_format.ParseDict(
                self.backend_config.tfdl_config, TrainingParametersMsg())
            log.info('tfdl_config={}'.format(tfdl_config))
            log.info('Training steps={}'.format(
                tfdl_config.training_number_of_steps))

            # Additional training options
            max_class = max(
                list(map(lambda c: c.id, self.class_map.get_items())))
            num_classes = len(self.class_map.get_items())
            num_classes = max(max_class, num_classes) + 1
            (train_args, train_env) = get_training_args(
                train_py, train_logdir_local, tfic_ckpt, dataset_dir_local,
                num_classes, tfdl_config)

            # Start training
            log.info('Starting training process')
            train_process = Popen(train_args, env=train_env)
            terminate_at_exit(train_process)

            if self.backend_config.train_options.do_monitoring:
                # Start tensorboard
                log.info('Starting tensorboard process')
                tensorboard_process = Popen(
                    ['tensorboard', '--logdir={}'.format(train_logdir_local)])
                terminate_at_exit(tensorboard_process)

            # Wait for training and tensorboard
            log.info('Waiting for training and tensorboard processes')
            train_process.wait()
            if self.backend_config.train_options.do_monitoring:
                tensorboard_process.terminate()

            # Export frozen graph
            log.info(
                'Exporting frozen graph ({}/model)'.format(train_logdir_local))
            export_args = get_export_args(export_py, train_logdir_local,
                                          num_classes, tfdl_config)
            export_process = Popen(export_args)
            terminate_at_exit(export_process)
            export_process.wait()

            # Package up the model files for usage as fine tuning checkpoints
            fine_tune_checkpoint_name = self.backend_config.fine_tune_checkpoint_name
            latest_checkpoints = get_latest_checkpoint(train_logdir_local)
            model_checkpoint_files = glob.glob(
                '{}*'.format(latest_checkpoints))
            inference_graph_path = os.path.join(train_logdir_local, 'model')

            with RVConfig.get_tmp_dir() as tmp_dir:
                model_dir = os.path.join(tmp_dir, fine_tune_checkpoint_name)
                make_dir(model_dir)
                model_tar = os.path.join(
                    train_logdir_local,
                    '{}.tar.gz'.format(fine_tune_checkpoint_name))
                shutil.copy(inference_graph_path,
                            '{}/frozen_inference_graph.pb'.format(model_dir))
                for path in model_checkpoint_files:
                    shutil.copy(path, model_dir)
                with tarfile.open(model_tar, 'w:gz') as tar:
                    tar.add(model_dir, arcname=os.path.basename(model_dir))

        # Perform final sync
        sync_to_dir(train_logdir_local, train_logdir, delete=False)

    def load_model(self, tmp_dir: str):
        """Load the model in preparation for one or more prediction calls.

        Args:
             tmp_dir: (str) temporary directory to use
        """
        # noqa Courtesy of https://github.com/tensorflow/models/blob/cbbb2ffcde66e646d4a47628ffe2ece2322b64e8/research/deeplab/deeplab_demo.ipynb
        import tensorflow as tf
        if self.sess is None:
            FROZEN_GRAPH_NAME = download_if_needed(
                self.backend_config.model_uri, tmp_dir)
            graph = tf.Graph()
            with open(FROZEN_GRAPH_NAME, 'rb') as data:
                graph_def = tf.GraphDef.FromString(data.read())
            with graph.as_default():
                tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session(graph=graph)

    def predict(self, chips: np.ndarray, windows: List[Box],
                tmp_dir: str) -> List[Tuple[Box, np.ndarray]]:
        """Predict using an already-trained DeepLab model.

        Args:
            chips: An np.ndarray containing the image data.
            windows: A list of windows corresponding to the respective
                 training chips.
             tmp_dir: (str) temporary directory to use
        Returns:
             A list of Box × np.ndarray pairs.

        """
        self.load_model(tmp_dir)
        labels = SemanticSegmentationLabels()

        # Feeding in one chip at a time because the model doesn't seem to
        # accept > 1.
        # TODO fix this
        for ind, window in enumerate(windows):
            class_labels = self.sess.run(
                OUTPUT_TENSOR_NAME,
                feed_dict={INPUT_TENSOR_NAME: [chips[ind]]})[0]
            labels.add_label_pair(window, class_labels)

        return labels
