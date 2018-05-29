import io
import tempfile
import os
import shutil
import zipfile
import tarfile
from os.path import join
from urllib.parse import urlparse
from subprocess import Popen
import signal
import atexit
import glob
import re
import uuid

from PIL import Image
import tensorflow as tf
import numpy as np
from google.protobuf import text_format

from object_detection.utils import dataset_util
from object_detection.utils.np_box_list import BoxList
from object_detection.utils.np_box_list_ops import scale
from object_detection.protos.string_int_label_map_pb2 import (
    StringIntLabelMap, StringIntLabelMapItem)
from object_detection.protos.pipeline_pb2 import TrainEvalPipelineConfig

from rastervision.core.ml_backend import MLBackend
from rastervision.ml_tasks.object_detection import save_debug_image
from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels)
from rastervision.utils.files import (
    get_local_path, upload_if_needed, make_dir, download_if_needed,
    file_to_str, sync_dir, start_sync)

TRAIN = 'train'
VALIDATION = 'validation'


def create_tf_example(image, labels, class_map, chip_id=''):
    image = Image.fromarray(image)
    image_format = 'png'
    encoded_image = io.BytesIO()
    image.save(encoded_image, format=image_format)
    width, height = image.size

    ymins, xmins, ymaxs, xmaxs = labels.get_coordinates()
    class_ids = labels.get_class_ids()
    class_names = [class_map.get_by_id(class_id).name.encode('utf8')
                   for class_id in class_ids]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(chip_id.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image.getvalue()),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(
            class_names),
        'image/object/class/label': dataset_util.int64_list_feature(
            class_ids)
    }))

    return tf_example


def write_tf_record(tf_examples, output_path):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())


def merge_tf_records(output_path, src_records):
    with tf.python_io.TFRecordWriter(output_path) as writer:
        print('Merging TFRecords', end='', flush=True)
        for src_record in src_records:
            for string_record in tf.python_io.tf_record_iterator(src_record):
                writer.write(string_record)
            print('.', end='', flush=True)
        print()


def make_tf_class_map(class_map):
    tf_class_map = StringIntLabelMap()
    tf_items = []
    for class_item in class_map.get_items():
        tf_item = StringIntLabelMapItem(id=class_item.id, name=class_item.name)
        tf_items.append(tf_item)
    tf_class_map.item.extend(tf_items)
    return tf_class_map


def save_tf_class_map(tf_class_map, class_map_path):
    tf_class_map_str = text_format.MessageToString(tf_class_map)
    with open(class_map_path, 'w') as class_map_file:
        class_map_file.write(tf_class_map_str)


def make_tf_examples(training_data, class_map):
    tf_examples = []
    print('Creating TFRecord', end='', flush=True)
    for chip, labels in training_data:
        tf_example = create_tf_example(chip, labels, class_map)
        tf_examples.append(tf_example)
        print('.', end='', flush=True)
    print()
    return tf_examples


def parse_tfexample(example):
    # Parse image.
    im_str = example.features.feature['image/encoded'].bytes_list.value[0]
    im = Image.open(io.BytesIO(im_str))
    im = np.asarray(im, dtype=np.uint8).copy()

    # Parse labels.
    xmins = example.features.feature['image/object/bbox/xmin'].float_list.value
    ymins = example.features.feature['image/object/bbox/ymin'].float_list.value
    xmaxs = example.features.feature['image/object/bbox/xmax'].float_list.value
    ymaxs = example.features.feature['image/object/bbox/ymax'].float_list.value

    nb_boxes = len(xmins)
    npboxes = np.empty((nb_boxes, 4))
    npboxes[:, 0] = ymins
    npboxes[:, 1] = xmins
    npboxes[:, 2] = ymaxs
    npboxes[:, 3] = xmaxs

    class_ids = example.features.feature[
        'image/object/class/label'].int64_list.value
    class_ids = np.array(class_ids)

    labels = ObjectDetectionLabels(npboxes, class_ids)
    return im, labels


def make_debug_images(record_path, class_map, output_dir):
    make_dir(output_dir, check_empty=True)

    print('Generating debug chips', end='', flush=True)
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        example = tf.train.Example.FromString(example)
        im, labels = parse_tfexample(example)
        output_path = join(output_dir, '{}.png'.format(ind))
        save_debug_image(im, labels, class_map, output_path)
        print('.', end='', flush=True)
    print()


def terminate_at_exit(process):
    def terminate():
        print('Terminating {}...'.format(process.pid))
        process.terminate()
    atexit.register(terminate)


def train(config_path, output_dir, train_py=None, eval_py=None):
    output_train_dir = join(output_dir, 'train')
    output_eval_dir = join(output_dir, 'eval')

    train_py = train_py or '/opt/tf-models/object_detection/train.py'
    eval_py = eval_py or '/opt/tf-models/object_detection/eval.py'

    train_process = Popen([
        'python', train_py,
        '--logtostderr', '--pipeline_config_path={}'.format(config_path),
        '--train_dir={}'.format(output_train_dir)])
    terminate_at_exit(train_process)

    eval_process = Popen([
        'python', eval_py,
        '--logtostderr', '--pipeline_config_path={}'.format(config_path),
        '--checkpoint_dir={}'.format(output_train_dir),
        '--eval_dir={}'.format(output_eval_dir)])
    terminate_at_exit(eval_process)

    tensorboard_process = Popen([
        'tensorboard', '--logdir={}'.format(output_dir)])
    terminate_at_exit(tensorboard_process)

    train_process.wait()
    eval_process.terminate()
    tensorboard_process.terminate()


def get_last_checkpoint_path(train_root_dir):
    index_paths = glob.glob(join(train_root_dir, 'train', '*.index'))
    checkpoint_ids = []
    for index_path in index_paths:
        match = re.match(r'model.ckpt-(\d+).index',
                         os.path.basename(index_path))
        checkpoint_ids.append(int(match.group(1)))

    if len(checkpoint_ids) == 0:
        return None
    checkpoint_id = max(checkpoint_ids)
    checkpoint_path = join(
        train_root_dir, 'train', 'model.ckpt-{}'.format(checkpoint_id))
    return checkpoint_path


def export_inference_graph(
    train_root_dir, config_path, output_dir, export_py=None):
    export_py = export_py or '/opt/tf-models/object_detection/export_inference_graph.py'
    checkpoint_path = get_last_checkpoint_path(train_root_dir)
    if checkpoint_path is None:
        print('No checkpoints could be found.')
    else:
        print('Exporting checkpoint {}...'.format(checkpoint_path))

        train_process = Popen([
            'python', export_py,
            '--input_type', 'image_tensor',
            '--pipeline_config_path', config_path,
            '--trained_checkpoint_prefix', checkpoint_path,
            '--output_directory', output_dir])
        train_process.wait()

        # Move frozen inference graph and clean up generated files.
        inference_graph_path = join(output_dir, 'frozen_inference_graph.pb')
        output_path = join(output_dir, 'model')
        shutil.move(inference_graph_path, output_path)
        saved_model_dir = join(output_dir, 'saved_model')
        shutil.rmtree(saved_model_dir)


class TrainingPackage(object):

    def __init__(self, base_uri):
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name

        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)
        make_dir(self.base_dir)

    def get_local_path(self, uri):
        return get_local_path(uri, self.temp_dir)

    def upload_if_needed(self, uri):
        upload_if_needed(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.temp_dir)

    def get_record_uri(self, split):
        return join(self.base_uri, '{}.record'.format(split))

    def get_debug_chips_uri(self, split):
        return join(self.base_uri, '{}-debug-chips.zip'.format(split))

    def get_class_map_uri(self):
        return join(self.base_uri, 'label-map.pbtxt')

    def upload(self, debug=False):
        self.upload_if_needed(self.get_record_uri(TRAIN))
        self.upload_if_needed(self.get_record_uri(VALIDATION))
        self.upload_if_needed(self.get_class_map_uri())
        if debug:
            self.upload_if_needed(self.get_debug_chips_uri(TRAIN))
            self.upload_if_needed(self.get_debug_chips_uri(VALIDATION))

    def download_data(self):
        # No need to download debug chips.
        self.download_if_needed(self.get_record_uri(TRAIN))
        self.download_if_needed(self.get_record_uri(VALIDATION))
        self.download_if_needed(self.get_class_map_uri())

    def download_pretrained_model(self, pretrained_model_zip_uri):
        # Expected to be .tar.gz file downloaded from
        # https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md # noqa
        pretrained_model_zip_path = self.download_if_needed(
            pretrained_model_zip_uri)
        pretrained_model_dir = join(self.temp_dir, 'pretrained_model')
        make_dir(pretrained_model_dir)
        with tarfile.open(pretrained_model_zip_path, 'r:gz') as tar:
            tar.extractall(pretrained_model_dir)
        model_name = os.path.splitext(os.path.splitext(
            os.path.basename(pretrained_model_zip_uri))[0])[0]
        # The unzipped file is assumed to have a single directory with
        # the name of the model derived from the zip file.
        pretrained_model_path = join(
            pretrained_model_dir, model_name, 'model.ckpt')
        return pretrained_model_path

    def download_config(self, pretrained_model_zip_uri, backend_config_uri):
        # Download and parse config.
        config_str = file_to_str(backend_config_uri)
        config = text_format.Parse(config_str, TrainEvalPipelineConfig())

        # Update config using local paths.
        pretrained_model_path = self.download_pretrained_model(
            pretrained_model_zip_uri)
        config.train_config.fine_tune_checkpoint = pretrained_model_path

        class_map_path = self.get_local_path(self.get_class_map_uri())

        train_path = self.get_local_path(self.get_record_uri(TRAIN))
        if hasattr(config.train_input_reader.tf_record_input_reader.input_path,
                   'append'):
            config.train_input_reader.tf_record_input_reader.input_path[:] = \
                [train_path]
        else:
            config.train_input_reader.tf_record_input_reader.input_path = \
                train_path
        config.train_input_reader.label_map_path = class_map_path

        eval_path = self.get_local_path(self.get_record_uri(VALIDATION))
        if hasattr(config.eval_input_reader.tf_record_input_reader.input_path,
                   'append'):
            config.eval_input_reader.tf_record_input_reader.input_path[:] = \
                [eval_path]
        else:
            config.eval_input_reader.tf_record_input_reader.input_path = \
                eval_path
        config.eval_input_reader.label_map_path = class_map_path

        # Save an updated copy of the config file.
        config_path = join(self.temp_dir, 'ml.config')
        config_str = text_format.MessageToString(config)
        with open(config_path, 'w') as config_file:
            config_file.write(config_str)
        return config_path


def load_frozen_graph(inference_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def compute_prediction(image_np, detection_graph, session):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    class_ids = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    (boxes, scores, class_ids) = session.run(
        [boxes, scores, class_ids],
        feed_dict={image_tensor: image_np_expanded})
    npboxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    class_ids = np.squeeze(class_ids).astype(np.int32)

    return ObjectDetectionLabels(npboxes, class_ids, scores=scores)


class TFObjectDetectionAPI(MLBackend):

    def __init__(self):
        self.detection_graph = None
        # persist scene training packages for when output_uri is remote
        self.scene_training_packages = []

    def process_scene_data(self, scene, data, class_map, options):
        """Process each scene's training data

        Args:
            scene: Scene
            data: TrainingData
            class_map: ClassMap
            options: MakeTrainingChipsConfig.Options

        Returns:
            the local path to the scene's TFRecord
        """

        training_package = TrainingPackage(options.output_uri)
        self.scene_training_packages.append(training_package)
        tf_examples = make_tf_examples(data, class_map)
        # Ensure directory is unique since scene id's could be shared between
        # training and test sets.
        record_path = training_package.get_local_path(
            training_package.get_record_uri('{}-{}'.format(
                scene.id, uuid.uuid4())))
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results, validation_results,
                                 class_map, options):
        """After all scenes have been processed, merge all TFRecords

        Args:
            training_results: list of training scenes' TFRecords
            validation_results: list of validation scenes' TFRecords
            class_map: ClassMap
            options: MakeTrainingChipsConfig.Options
        """

        training_package = TrainingPackage(options.output_uri)

        def _merge_training_results(results, split):

            # "split" tf record
            record_path = training_package.get_local_path(
                training_package.get_record_uri(split))

            # merge each scene's tfrecord into "split" tf record
            merge_tf_records(record_path, results)

            # Save debug chips.
            if options.debug:
                debug_zip_path = training_package.get_local_path(
                    training_package.get_debug_chips_uri(split))
                with tempfile.TemporaryDirectory() as debug_dir:
                    make_debug_images(record_path, class_map, debug_dir)
                    shutil.make_archive(
                        os.path.splitext(debug_zip_path)[0], 'zip', debug_dir)

        _merge_training_results(training_results, TRAIN)
        _merge_training_results(validation_results, VALIDATION)

        # Save TF label map based on class_map.
        class_map_path = training_package.get_local_path(
            training_package.get_class_map_uri())
        tf_class_map = make_tf_class_map(class_map)
        save_tf_class_map(tf_class_map, class_map_path)

        training_package.upload(debug=options.debug)

        # clear scene training packages
        del self.scene_training_packages[:]

    def train(self, class_map, options):
        # Download training data and update config file.
        training_package = TrainingPackage(options.training_data_uri)
        training_package.download_data()
        config_path = training_package.download_config(
            options.pretrained_model_uri, options.backend_config_uri)

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup output dirs.
            output_dir = get_local_path(options.output_uri, temp_dir)
            make_dir(output_dir)

            train_py = options.object_detection_options.train_py
            eval_py = options.object_detection_options.eval_py
            export_py = options.object_detection_options.export_py

            # Train model and sync output periodically.
            start_sync(output_dir, options.output_uri,
                       sync_interval=options.sync_interval)
            train(config_path, output_dir, train_py=train_py, eval_py=eval_py)

            export_inference_graph(
                output_dir, config_path, output_dir,
                export_py=export_py)

            if urlparse(options.output_uri).scheme == 's3':
                sync_dir(output_dir, options.output_uri, delete=True)

    def predict(self, chip, options):
        # Load and memoize the detection graph and TF session.
        if self.detection_graph is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                model_path = download_if_needed(options.model_uri, temp_dir)
                self.detection_graph = load_frozen_graph(model_path)
                self.session = tf.Session(graph=self.detection_graph)

        # If chip is blank, then return empty predictions.
        if np.sum(np.ravel(chip)) == 0:
            return ObjectDetectionLabels.make_empty()
        return compute_prediction(chip, self.detection_graph, self.session)
