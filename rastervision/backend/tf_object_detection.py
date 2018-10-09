import rastervision as rv

import io
import os
import shutil
import tarfile
from os.path import join
from subprocess import (Popen, PIPE, STDOUT)
import glob
import re
import uuid
import logging
from copy import deepcopy

from PIL import Image
import numpy as np
from google.protobuf import text_format, json_format

from rastervision.backend import Backend
from rastervision.data import ObjectDetectionLabels
from rastervision.utils.files import (get_local_path, upload_or_copy, make_dir,
                                      download_if_needed, sync_to_dir,
                                      sync_from_dir, start_sync)
from rastervision.utils.misc import (save_img, replace_nones_in_dict,
                                     terminate_at_exit)
from rastervision.rv_config import RVConfig

TRAIN = 'train'
VALIDATION = 'validation'

log = logging.getLogger(__name__)


def save_debug_image(im, labels, class_map, output_path):
    from object_detection.utils import visualization_utils as vis_util

    npboxes = labels.get_npboxes()
    class_ids = labels.get_class_ids()
    scores = labels.get_scores()
    if scores is None:
        scores = [1.0] * len(labels)

    vis_util.visualize_boxes_and_labels_on_image_array(
        im,
        npboxes,
        class_ids,
        scores,
        class_map.get_category_index(),
        use_normalized_coordinates=True,
        line_thickness=2,
        max_boxes_to_draw=None)
    save_img(im, output_path)


def create_tf_example(image, window, labels, class_map, chip_id=''):
    import tensorflow as tf
    from object_detection.utils import dataset_util

    image = Image.fromarray(image)
    image_format = 'png'
    encoded_image = io.BytesIO()
    image.save(encoded_image, format=image_format)
    width, height = image.size

    npboxes = labels.get_npboxes()
    npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
    npboxes = ObjectDetectionLabels.local_to_normalized(npboxes, window)
    ymins = npboxes[:, 0]
    xmins = npboxes[:, 1]
    ymaxs = npboxes[:, 2]
    xmaxs = npboxes[:, 3]
    class_ids = labels.get_class_ids()
    class_names = [
        class_map.get_by_id(class_id).name.encode('utf8')
        for class_id in class_ids
    ]

    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                dataset_util.int64_feature(height),
                'image/width':
                dataset_util.int64_feature(width),
                'image/filename':
                dataset_util.bytes_feature(chip_id.encode('utf8')),
                'image/source_id':
                dataset_util.bytes_feature(chip_id.encode('utf8')),
                'image/encoded':
                dataset_util.bytes_feature(encoded_image.getvalue()),
                'image/format':
                dataset_util.bytes_feature(image_format.encode('utf8')),
                'image/object/bbox/xmin':
                dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax':
                dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin':
                dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax':
                dataset_util.float_list_feature(ymaxs),
                'image/object/class/text':
                dataset_util.bytes_list_feature(class_names),
                'image/object/class/label':
                dataset_util.int64_list_feature(class_ids)
            }))

    return tf_example


def write_tf_record(tf_examples, output_path: str) -> None:
    """Write an array of TFRecords to the given output path.

    Args:
         tf_examples: An array of TFRecords; a
              list(tensorflow.core.example.example_pb2.Example)
         output_path: The path where the records should be stored.

    Returns:
         None

    """
    import tensorflow as tf

    with tf.python_io.TFRecordWriter(output_path) as writer:
        for tf_example in tf_examples:
            writer.write(tf_example.SerializeToString())


def merge_tf_records(output_path, src_records):
    import tensorflow as tf

    with tf.python_io.TFRecordWriter(output_path) as writer:
        log.info('Merging TFRecords')
        for src_record in src_records:
            for string_record in tf.python_io.tf_record_iterator(src_record):
                writer.write(string_record)


def make_tf_class_map(class_map):
    from rastervision.protos.tf_object_detection.string_int_label_map_pb2 import (
        StringIntLabelMap, StringIntLabelMapItem)

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
    log.info('Creating TFRecord')
    for chip, window, labels in training_data:
        tf_example = create_tf_example(chip, window, labels, class_map)
        tf_examples.append(tf_example)
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
    import tensorflow as tf

    make_dir(output_dir, check_empty=True)

    log.info('Generating debug chips')
    tfrecord_iter = tf.python_io.tf_record_iterator(record_path)
    for ind, example in enumerate(tfrecord_iter):
        example = tf.train.Example.FromString(example)
        im, labels = parse_tfexample(example)
        # Can't create debug images for non-3band images
        if im.shape[2] != 3:
            log.warning(
                'WARNING: Skipping debug images - Images are not 3 band rasters.'
            )
            return
        output_path = join(output_dir, '{}.png'.format(ind))
        save_debug_image(im, labels, class_map, output_path)


def train(config_path,
          output_dir,
          num_steps,
          model_main_py=None,
          do_monitoring=True):
    output_train_dir = join(output_dir, 'train')
    output_eval_dir = join(output_dir, 'eval')

    model_main_py = model_main_py or '/opt/tf-models/object_detection/model_main.py'

    train_cmd = [
        'python', model_main_py, '--alsologtostderr',
        '--pipeline_config_path={}'.format(config_path),
        '--model_dir={}'.format(output_train_dir),
        '--num_train_steps={}'.format(num_steps),
        '--sample_1_of_n_eval_examples={}'.format(1)
    ]

    log.info('Running train command: {}'.format(' '.join(train_cmd)))

    train_process = Popen(train_cmd, stdout=PIPE, stderr=STDOUT)
    terminate_at_exit(train_process)

    if do_monitoring:
        eval_cmd = [
            'python', model_main_py, '--alsologtostderr',
            '--pipeline_config_path={}'.format(config_path),
            '--checkpoint_dir={}'.format(output_train_dir),
            '--model_dir={}'.format(output_eval_dir)
        ]
        log.info('Running eval command: {}'.format(' '.join(eval_cmd)))

        # Don't let the eval process take up GPU space
        env = deepcopy(os.environ)
        env['CUDA_VISIBLE_DEVICES'] = '-1'
        eval_process = Popen(eval_cmd, env=env)

        tensorboard_process = Popen(
            ['tensorboard', '--logdir={}'.format(output_dir)])
        terminate_at_exit(eval_process)
        terminate_at_exit(tensorboard_process)

    with train_process:
        for line in train_process.stdout:
            log.info(line.decode('utf-8'))

    log.info('-----DONE TRAINING----')
    if do_monitoring:
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
    checkpoint_path = join(train_root_dir, 'train',
                           'model.ckpt-{}'.format(checkpoint_id))
    return checkpoint_path


def export_inference_graph(train_root_dir,
                           config_path,
                           output_dir,
                           fine_tune_checkpoint_name,
                           export_py=None):
    export_py = (export_py or
                 '/opt/tf-models/object_detection/export_inference_graph.py')
    checkpoint_path = get_last_checkpoint_path(train_root_dir)
    if checkpoint_path is None:
        log.warning('No checkpoints could be found.')
    else:
        log.info('Exporting checkpoint {}...'.format(checkpoint_path))

        train_process = Popen([
            'python', export_py, '--input_type', 'image_tensor',
            '--pipeline_config_path', config_path,
            '--trained_checkpoint_prefix', checkpoint_path,
            '--output_directory', output_dir
        ])
        train_process.wait()

        inference_graph_path = join(output_dir, 'frozen_inference_graph.pb')

        # Package up the model files for usage as fine tuning checkpoints
        model_checkpoint_files = [
            os.path.join(output_dir, fname) for fname in os.listdir(output_dir)
            if fname.startswith('model.ckpt')
        ]
        with RVConfig.get_tmp_dir() as tmp_dir:
            model_dir = os.path.join(tmp_dir, fine_tune_checkpoint_name)
            make_dir(model_dir)
            model_tar = os.path.join(
                output_dir, '{}.tar.gz'.format(fine_tune_checkpoint_name))
            shutil.copy(inference_graph_path, model_dir)
            for path in model_checkpoint_files:
                shutil.copy(path, model_dir)
            with tarfile.open(model_tar, 'w:gz') as tar:
                tar.add(model_dir, arcname=os.path.basename(model_dir))

        # Move frozen inference graph and clean up generated files.

        output_path = join(output_dir, 'model')
        shutil.move(inference_graph_path, output_path)
        saved_model_dir = join(output_dir, 'saved_model')
        shutil.rmtree(saved_model_dir)


class TrainingPackage(object):
    """Encapsulates the files needed to train a model.

    This encapsulates the files that are generated by the make_chips
    command and that are used by the train command. It takes the URI of the
    directory used to store these files (which is remote or local depending on
    where the command is being executed), generates the URIs of all the
    individual files, and downloads and uploads them. This assumes the
    directory has the following structure:
        label-map.pbtxt
        train-debug-chips.zip
        train.record
        validation-debug-chips.zip
        validation.record
    """

    def __init__(self, base_uri, config, temp_dir):
        """Constructor.

        Creates a temporary directory.

        Args:
            base_uri: (string) URI of directory containing files used to
                train model, possibly remote
            config: TFObjectDetectionConfig
        """

        self.temp_dir = temp_dir

        self.base_uri = base_uri
        self.base_dir = self.get_local_path(base_uri)

        make_dir(self.base_dir)

        self.config = config

    def get_local_path(self, uri):
        """Get local version of uri.

        Args:
            uri: (string) URI of file, possibly remote

        Returns:
            (string) path of local version of file
        """
        return get_local_path(uri, self.temp_dir)

    def upload_or_copy(self, uri):
        """Upload file if it's remote.

        This knows how to generate the path to the local copy of the file.

        Args:
            uri: (string) URI of file, possibly remote
        """
        upload_or_copy(self.get_local_path(uri), uri)

    def download_if_needed(self, uri):
        """Download file if it's remote.

        Args:
            uri: (string) URI of file, possibly remote

        Returns:
            (string) path of local file that was downloaded
        """
        return download_if_needed(uri, self.temp_dir)

    def get_record_uri(self, split):
        """Get URI of TFRecord for dataset split.

        Args:
            split: (string) 'train' or 'validation'

        Returns:
            (string) URI of TFRecord file, possibly remote
        """
        return join(self.base_uri, '{}.record'.format(split))

    def get_debug_chips_uri(self, split):
        """Get URI of debug chips zip file for dataset split.

        Args:
            split: (string) 'train' or 'validation'

        Returns:
            (string) URI of zip file containing debug chips, possibly remote
        """
        return join(self.base_uri, '{}-debug-chips.zip'.format(split))

    def get_class_map_uri(self):
        """Get URI of class map file.

        The TF Object Detection API uses its own class map file that maps
        class_ids to class_names, which they call a "label map".

        Returns:
            (string) URI of class map file, possibly remote
        """
        return join(self.base_uri, 'label-map.pbtxt')

    def upload(self, debug=False):
        """Upload training and validation data, and class map files.

        Args:
            debug: (bool) if True, also upload the corresponding debug chip
                zip files
        """
        self.upload_or_copy(self.get_record_uri(TRAIN))
        self.upload_or_copy(self.get_record_uri(VALIDATION))
        self.upload_or_copy(self.get_class_map_uri())
        if debug:
            self.upload_or_copy(self.get_debug_chips_uri(TRAIN))
            self.upload_or_copy(self.get_debug_chips_uri(VALIDATION))

    def download_data(self):
        """Download training and validation data, and class map files."""
        # No need to download debug chips.
        self.download_if_needed(self.get_record_uri(TRAIN))
        self.download_if_needed(self.get_record_uri(VALIDATION))
        self.download_if_needed(self.get_class_map_uri())

    def download_pretrained_model(self, pretrained_model_zip_uri):
        """Download pretrained model and unzip it.

        This is used before training a model.

        Args:
            pretrained_model_zip_uri: (string) URI of .tar.gz file containing
                pretrained model. This file is of the form that comes from the
                Model Zoo at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md # noqa
                It contains a directory at the top level with the same name as
                root of the zip file (if zip file is x.tar.gz, the directory
                is x), and a set of files of the form model.ckpt.*. This file
                can be located anywhere, and is not expected to be in the
                directory encapsulated by this class that is generated by the
                make_chips command. That is why it is passed in
                separately.

        Returns:
            (string) path to pretrained model file (which is model.ckpt in
                the zip file)
        """
        pretrained_model_zip_path = self.download_if_needed(
            pretrained_model_zip_uri)
        pretrained_model_dir = join(self.temp_dir, 'pretrained_model')
        make_dir(pretrained_model_dir)
        with tarfile.open(pretrained_model_zip_path, 'r:gz') as tar:
            tar.extractall(pretrained_model_dir)
        model_name = os.path.splitext(
            os.path.splitext(os.path.basename(pretrained_model_zip_uri))[0])[0]
        # The unzipped file is assumed to have a single directory with
        # the name of the model derived from the zip file.
        pretrained_model_path = join(pretrained_model_dir, model_name,
                                     'model.ckpt')
        return pretrained_model_path

    def download_config(self):
        from rastervision.protos.tf_object_detection.pipeline_pb2 \
            import TrainEvalPipelineConfig
        """Download a model and backend config and update its fields.

        This is used before training a model. This downloads and unzips a bunch
        of files that are needed to train a model, and then downloads and
        updates the backend config file with local paths to these files. These
        files include the pretrained model, the class map, and the training and
        validation datasets.

        Args:
            pretrained_model_zip_uri: (string) URI of .tar.gz file containing
                pretrained model. (See download_pretrained_model method for more
                details.)
            backend_config_uri: (string) URI of backend config file which is
                a config file for the TF Object Detection API. Examples can be
                found here https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs  # noqa
        """

        # Parse configuration
        # We must remove 'nulls' that appear due to translating empty
        # messages. These appear when translating between text and JSON based
        # protobuf messages, and using the google.protobuf.Struct type to store
        # the JSON. This appears when TFOD uses empty message types as an enum.
        config = json_format.ParseDict(
            replace_nones_in_dict(self.config.tfod_config, {}),
            TrainEvalPipelineConfig())

        # Update config using local paths.
        if config.train_config.fine_tune_checkpoint:
            pretrained_model_path = self.download_pretrained_model(
                config.train_config.fine_tune_checkpoint)
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

        if hasattr(
                config.eval_input_reader[0].tf_record_input_reader.input_path,
                'append'):
            config.eval_input_reader[0].tf_record_input_reader.input_path[:] = \
                [eval_path]
        else:
            config.eval_input_reader[0].tf_record_input_reader.input_path = \
                eval_path
        config.eval_input_reader[0].label_map_path = class_map_path

        # Save an updated copy of the config file.
        config_path = join(self.temp_dir, 'ml.config')
        config_str = text_format.MessageToString(config)
        with open(config_path, 'w') as config_file:
            config_file.write(config_str)
        return config_path


def load_frozen_graph(inference_graph_path):
    import tensorflow as tf

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def compute_prediction(image_nps, windows, detection_graph, session):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    class_ids = detection_graph.get_tensor_by_name('detection_classes:0')

    (boxes, scores, class_ids) = session.run(
        [boxes, scores, class_ids], feed_dict={image_tensor: image_nps})

    labels = ObjectDetectionLabels.make_empty()
    for chip_boxes, chip_scores, chip_class_ids, window in zip(
            boxes, scores, class_ids, windows):
        chip_boxes = ObjectDetectionLabels.normalized_to_local(
            chip_boxes, window)
        chip_boxes = ObjectDetectionLabels.local_to_global(chip_boxes, window)
        chip_class_ids = chip_class_ids.astype(np.int32)
        labels = (labels + ObjectDetectionLabels(
            chip_boxes, chip_class_ids, scores=chip_scores))

    return labels


class TFObjectDetection(Backend):
    def __init__(self, backend_config, task_config):
        self.detection_graph = None
        # persist scene training packages for when output_uri is remote
        self.scene_training_packages = []
        self.config = backend_config
        self.class_map = task_config.class_map

    def process_scene_data(self, scene, data, tmp_dir):
        """Process each scene's training data

        Args:
            scene: Scene
            data: TrainingData
            class_map: ClassMap

        Returns:
            the local path to the scene's TFRecord
        """
        # Currently TF Object Detection can only handle uint8
        if scene.raster_source.get_dtype() != np.uint8:
            raise Exception('Cannot use {} backend for imagery that does '
                            'not have data type uint8. '
                            'Use the StatsAnalyzer and StatsTransformer '
                            'to turn the raster data into uint8 data'.format(
                                rv.TF_OBJECT_DETECTION))

        training_package = TrainingPackage(self.config.training_data_uri,
                                           self.config, tmp_dir)
        self.scene_training_packages.append(training_package)
        tf_examples = make_tf_examples(data, self.class_map)
        # Ensure directory is unique since scene id's could be shared between
        # training and test sets.
        record_path = training_package.get_local_path(
            training_package.get_record_uri('{}-{}'.format(
                scene.id, uuid.uuid4())))
        record_path = training_package.get_local_path(
            training_package.get_record_uri('{}-{}'.format(
                scene.id, uuid.uuid4())))
        write_tf_record(tf_examples, record_path)

        return record_path

    def process_sceneset_results(self, training_results, validation_results,
                                 tmp_dir):
        """After all scenes have been processed, merge all TFRecords

        Args:
            training_results: list of training scenes' TFRecords
            validation_results: list of validation scenes' TFRecords
        """
        training_package = TrainingPackage(self.config.training_data_uri,
                                           self.config, tmp_dir)

        def _merge_training_results(results, split):

            # "split" tf record
            record_path = training_package.get_local_path(
                training_package.get_record_uri(split))

            # merge each scene's tfrecord into "split" tf record
            merge_tf_records(record_path, results)

            # Save debug chips.
            if self.config.debug:
                debug_zip_path = training_package.get_local_path(
                    training_package.get_debug_chips_uri(split))
                with RVConfig.get_tmp_dir() as debug_dir:
                    make_debug_images(record_path, self.class_map, debug_dir)
                    shutil.make_archive(
                        os.path.splitext(debug_zip_path)[0], 'zip', debug_dir)

        _merge_training_results(training_results, TRAIN)
        _merge_training_results(validation_results, VALIDATION)

        # Save TF label map based on class_map.
        class_map_path = training_package.get_local_path(
            training_package.get_class_map_uri())
        tf_class_map = make_tf_class_map(self.class_map)
        save_tf_class_map(tf_class_map, class_map_path)

        training_package.upload(debug=self.config.debug)

        # clear scene training packages
        del self.scene_training_packages[:]

    def train(self, tmp_dir):
        # Download training data and update config file.
        training_package = TrainingPackage(self.config.training_data_uri,
                                           self.config, tmp_dir)
        training_package.download_data()
        config_path = training_package.download_config()

        # Setup output dirs.
        output_dir = get_local_path(self.config.training_output_uri, tmp_dir)

        # Get output from potential previous run so we can resume training.
        if not self.config.train_options.replace_model:
            make_dir(output_dir)
            sync_from_dir(self.config.training_output_uri, output_dir)
        else:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            make_dir(output_dir)

        local_config_path = os.path.join(output_dir, 'pipeline.config')
        shutil.copy(config_path, local_config_path)

        model_main_py = self.config.script_locations.model_main_uri
        export_py = self.config.script_locations.export_uri

        # Train model and sync output periodically.
        sync = start_sync(
            output_dir,
            self.config.training_output_uri,
            sync_interval=self.config.train_options.sync_interval)
        with sync:
            train(
                local_config_path,
                output_dir,
                self.config.get_num_steps(),
                model_main_py=model_main_py,
                do_monitoring=self.config.train_options.do_monitoring)

        export_inference_graph(
            output_dir,
            local_config_path,
            output_dir,
            fine_tune_checkpoint_name=self.config.fine_tune_checkpoint_name,
            export_py=export_py)

        # Perform final sync
        sync_to_dir(output_dir, self.config.training_output_uri)

    def load_model(self, tmp_dir):
        import tensorflow as tf

        # Load and memoize the detection graph and TF session.
        if self.detection_graph is None:
            model_path = download_if_needed(self.config.model_uri, tmp_dir)
            self.detection_graph = load_frozen_graph(model_path)
            self.session = tf.Session(graph=self.detection_graph)

    def predict(self, chips, windows, tmp_dir):
        # Ensure model is loaded
        self.load_model(tmp_dir)

        return compute_prediction(chips, windows, self.detection_graph,
                                  self.session)
