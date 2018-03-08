import io
import tempfile
import os
import shutil
import zipfile
from os.path import join
from urllib.parse import urlparse
from subprocess import Popen
from threading import Timer
import signal
import atexit
import glob
import re

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

from rv2.core.ml_backend import MLBackend
from rv2.ml_methods.object_detection import save_debug_image
from rv2.annotations.object_detection_annotations import (
    ObjectDetectionAnnotations)
from rv2.utils.files import (
    get_local_path, upload_if_needed, make_dir, download_if_needed,
    file_to_str, sync_dir, RV_TEMP_DIR)

TRAIN = 'train'
VALIDATION = 'validation'


def create_tf_example(image, annotations, label_map, chip_id=''):
    image = Image.fromarray(image)
    image_format = 'png'
    encoded_image = io.BytesIO()
    image.save(encoded_image, format=image_format)
    width, height = image.size

    ymins, xmins, ymaxs, xmaxs = annotations.get_coordinates()
    classes = annotations.get_classes()
    class_texts = [label_map.get_by_id(class_id).name.encode('utf8')
                   for class_id in classes]

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
            class_texts),
        'image/object/class/label': dataset_util.int64_list_feature(
            classes)
    }))

    return tf_example


def write_tf_record(tf_examples, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    for tf_example in tf_examples:
        writer.write(tf_example.SerializeToString())
    writer.close()


def make_tf_label_map(label_map):
    tf_label_map = StringIntLabelMap()
    tf_items = []
    for label_item in label_map.get_items():
        tf_item = StringIntLabelMapItem(id=label_item.id, name=label_item.name)
        tf_items.append(tf_item)
    tf_label_map.item.extend(tf_items)
    return tf_label_map


def save_tf_label_map(tf_label_map, label_map_path):
    tf_label_map_str = text_format.MessageToString(tf_label_map)
    with open(label_map_path, 'w') as label_map_file:
        label_map_file.write(tf_label_map_str)


def make_tf_examples(train_data, label_map):
    tf_examples = []
    # TODO make train_data iterable
    print('Creating TFRecord', end='', flush=True)
    for chip, annotations in zip(train_data.chips, train_data.annotations):
        tf_example = create_tf_example(chip, annotations, label_map)
        tf_examples.append(tf_example)
        print('.', end='', flush=True)
    print()
    return tf_examples


def parse_tfexample(example):
    # Parse image.
    im_str = example.features.feature['image/encoded'].bytes_list.value[0]
    im = Image.open(io.BytesIO(im_str))
    im = np.asarray(im, dtype=np.uint8).copy()

    # Parse annotations.
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

    class_ids = example.features.feature['image/object/class/label'].int64_list.value
    class_ids = np.array(class_ids)

    annotations = ObjectDetectionAnnotations(npboxes, class_ids)
    return im, annotations


def make_debug_images(record_path, label_map, output_dir):
    make_dir(output_dir, check_empty=True)

    print('Generating debug chips', end='', flush=True)
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        example = tf.train.Example.FromString(example)
        im, annotations = parse_tfexample(example)
        output_path = join(output_dir, '{}.png'.format(ind))
        save_debug_image(im, annotations, label_map, output_path)
        print('.', end='', flush=True)
    print()


def terminate_at_exit(process):
    def terminate():
        print('Terminating {}...'.format(process.pid))
        process.terminate()
    atexit.register(terminate)


def train(config_path, output_dir):
    output_train_dir = join(output_dir, 'train')
    output_eval_dir = join(output_dir, 'eval')

    train_process = Popen([
        'python', '/opt/src/tf/object_detection/train.py',
        '--logtostderr', '--pipeline_config_path={}'.format(config_path),
        '--train_dir={}'.format(output_train_dir)])
    terminate_at_exit(train_process)

    eval_process = Popen([
        'python', '/opt/src/tf/object_detection/eval.py',
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


def start_sync(output_dir, output_uri, sync_interval=600):
    def sync_train_dir(delete=True):
        sync_dir(output_dir, output_uri, delete=delete)
        thread = Timer(sync_interval, sync_train_dir)
        thread.daemon = True
        thread.start()

    if urlparse(output_uri).scheme == 's3':
        # On first sync, we don't want to delete files on S3 to match
        # th contents of output_dir since there's nothing there yet.
        sync_train_dir(delete=False)


def get_last_checkpoint_path(train_root_dir):
    index_paths = glob.glob(join(train_root_dir, 'train', '*.index'))
    checkpoint_ids = []
    for index_path in index_paths:
        match = re.match(r'model.ckpt-(\d+).index', os.path.basename(index_path))
        checkpoint_ids.append(int(match.group(1)))

    if len(checkpoint_ids) == 0:
        return None
    checkpoint_id = max(checkpoint_ids)
    checkpoint_path = join(
        train_root_dir, 'train', 'model.ckpt-{}'.format(checkpoint_id))
    return checkpoint_path


def export_inference_graph(train_root_dir, config_path, inference_graph_path):
    checkpoint_path = get_last_checkpoint_path(train_root_dir)
    if checkpoint_path is None:
        print('No checkpoints could be found.')
    else:
        print('Exporting checkpoint {}...'.format(checkpoint_path))
        train_process = Popen([
            'python', '/opt/src/tf/object_detection/export_inference_graph.py',
            '--input_type', 'image_tensor',
            '--pipeline_config_path', config_path,
            '--checkpoint_path', checkpoint_path,
            '--inference_graph_path', inference_graph_path])
        train_process.wait()


class TrainPackage(object):
    def __init__(self, base_uri):
        self.temp_dir_obj = tempfile.TemporaryDirectory(dir=RV_TEMP_DIR)
        self.temp_dir = self.temp_dir_obj.name

        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)
        make_dir(self.base_dir)

    def get_record_uri(self, split):
        return join(self.base_uri, '{}.record'.format(split))

    def get_debug_chips_uri(self, split):
        return join(self.base_uri, '{}-debug-chips.zip'.format(split))

    def get_label_map_uri(self):
        return join(self.base_uri, 'label-map.pbtxt')

    def get_local_path(self, uri):
        return get_local_path(uri, self.temp_dir)

    def upload_if_needed(self, uri):
        upload_if_needed(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        return download_if_needed(uri, self.temp_dir)

    def upload(self, debug=False):
        self.upload_if_needed(self.get_record_uri(TRAIN))
        self.upload_if_needed(self.get_record_uri(VALIDATION))
        self.upload_if_needed(self.get_label_map_uri())
        if debug:
            self.upload_if_needed(self.get_debug_chips_uri(TRAIN))
            self.upload_if_needed(self.get_debug_chips_uri(VALIDATION))

    def download_data(self):
        # No need to download debug chips.
        self.download_if_needed(self.get_record_uri(TRAIN))
        self.download_if_needed(self.get_record_uri(VALIDATION))
        self.download_if_needed(self.get_label_map_uri())

    def download_pretrained_model(self, pretrained_model_zip_uri):
        pretrained_model_zip_path = self.download_if_needed(
            pretrained_model_zip_uri)
        zip_ref = zipfile.ZipFile(pretrained_model_zip_path, 'r')
        zip_ref.extractall(self.temp_dir)
        zip_ref.close()
        pretrained_model_path = join(self.temp_dir, 'model.ckpt')
        return pretrained_model_path

    def download_config(self, pretrained_model_zip_uri, backend_config_uri):
        # Download and parse config.
        config_str = file_to_str(backend_config_uri)
        config = text_format.Parse(config_str, TrainEvalPipelineConfig())

        # Update config using local paths.
        pretrained_model_path = self.download_pretrained_model(
            pretrained_model_zip_uri)
        config.train_config.fine_tune_checkpoint = pretrained_model_path

        label_map_path = self.get_local_path(self.get_label_map_uri())

        config.train_input_reader.tf_record_input_reader.input_path = \
            self.get_local_path(self.get_record_uri(TRAIN))
        config.train_input_reader.label_map_path = label_map_path

        config.eval_input_reader.tf_record_input_reader.input_path = \
            self.get_local_path(self.get_record_uri(VALIDATION))
        config.eval_input_reader.label_map_path = label_map_path

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
    # TODO predict by the batch-load
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name(
        'image_tensor:0')
    boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')

    (boxes, scores, classes) = session.run(
        [boxes, scores, classes],
        feed_dict={image_tensor: image_np_expanded})
    npboxes = np.squeeze(boxes)
    classes = np.squeeze(classes).astype(np.int32)
    scores = np.squeeze(scores)

    return ObjectDetectionAnnotations(npboxes, classes, scores=scores)


class TFObjectDetectionAPI(MLBackend):
    def __init__(self):
        self.detection_graph = None

    def convert_train_data(self, train_data, validation_data, label_map,
                           options):
        train_package = TrainPackage(options.output_uri)

        def _convert_train_data(data, split):
            # Save TFRecord.
            tf_examples = make_tf_examples(data, label_map)
            record_path = train_package.get_local_path(
                train_package.get_record_uri(split))
            write_tf_record(tf_examples, record_path)

            # Save debug chips.
            if options.debug:
                debug_zip_path = train_package.get_local_path(
                    train_package.get_debug_chips_uri(split))
                with tempfile.TemporaryDirectory(dir=RV_TEMP_DIR) as debug_dir:
                    make_debug_images(record_path, label_map, debug_dir)
                    shutil.make_archive(
                        os.path.splitext(debug_zip_path)[0], 'zip', debug_dir)

        _convert_train_data(train_data, TRAIN)
        _convert_train_data(validation_data, VALIDATION)

        # Save TF label map based on label_map.
        label_map_path = train_package.get_local_path(
            train_package.get_label_map_uri())
        tf_label_map = make_tf_label_map(label_map)
        save_tf_label_map(tf_label_map, label_map_path)

        train_package.upload(debug=options.debug)

    def train(self, options):
        # Download training data and update config file.
        train_package = TrainPackage(options.train_data_uri)
        train_package.download_data()
        config_path = train_package.download_config(
            options.pretrained_model_uri, options.backend_config_uri)

        with tempfile.TemporaryDirectory(dir=RV_TEMP_DIR) as temp_dir:
            # Setup output dirs.
            output_dir = get_local_path(options.output_uri, temp_dir)
            make_dir(output_dir)

            # Train model and sync output periodically.
            start_sync(output_dir, options.output_uri,
                       sync_interval=options.sync_interval)
            train(config_path, output_dir)

            # Export inference graph.
            inference_graph_path = join(output_dir, 'model')
            export_inference_graph(
                output_dir, config_path, inference_graph_path)

            if urlparse(options.output_uri).scheme == 's3':
                sync_dir(output_dir, options.output_uri, delete=True)

    def predict(self, chip, options):
        # Load and memoize the detection graph and TF session.
        if self.detection_graph is None:
            with tempfile.TemporaryDirectory(dir=RV_TEMP_DIR) as temp_dir:
                model_path = download_if_needed(options.model_uri, temp_dir)
                self.detection_graph = load_frozen_graph(model_path)
                self.session = tf.Session(graph=self.detection_graph)

        # If chip is blank, then return empty predictions.
        if np.sum(np.ravel(chip)) == 0:
            return ObjectDetectionAnnotations.make_empty()
        return compute_prediction(chip, self.detection_graph, self.session)
