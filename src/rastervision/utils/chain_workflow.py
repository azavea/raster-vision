from os.path import join, isfile
import copy
from urllib.parse import urlparse

import click
from google.protobuf.descriptor import FieldDescriptor
import boto3
import botocore

from rastervision.protos.chain_workflow_pb2 import ChainWorkflowConfig
from rastervision.protos.compute_raster_stats_pb2 import (
    ComputeRasterStatsConfig)
from rastervision.protos.make_training_chips_pb2 import (
    MakeTrainingChipsConfig)
from rastervision.protos.train_pb2 import TrainConfig
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.protos.eval_pb2 import EvalConfig
from rastervision.protos.label_store_pb2 import (
    LabelStore as LabelStoreConfig,
    ObjectDetectionGeoJSONFile as ObjectDetectionGeoJSONFileConfig,
    ClassificationGeoJSONFile as ClassificationGeoJSONFileConfig)

from rastervision.utils.files import (
    load_json_config, save_json_config, file_to_str, str_to_file)
from rastervision.utils.batch import _batch_submit
from rastervision import run

COMPUTE_RASTER_STATS = 'compute_raster_stats'
MAKE_TRAINING_CHIPS = 'make_training_chips'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_TASKS = [COMPUTE_RASTER_STATS, MAKE_TRAINING_CHIPS, TRAIN, PREDICT, EVAL]

validated_uri_fields = set([
    ('rv.protos.ObjectDetectionGeoJSONFile', 'uri'),
    ('rv.protos.ClassificationGeoJSONFile', 'uri'),
    ('rv.protos.GeoTiffFiles', 'uris'),
    ('rv.protos.ImageFile', 'uri'),
    ('rv.protos.TrainConfig.Options', 'backend_config_uri'),
    ('rv.protos.TrainConfig.Options', 'pretrained_model_uri')
])

s3 = boto3.resource('s3')


def make_command(command, config_uri):
    return 'python -m rastervision.run {} {}'.format(command, config_uri)


class PathGenerator(object):

    def __init__(self, uri_map, raw_dataset_key, dataset_key, model_key,
                 prediction_key, eval_key):
        rv_root = uri_map['rv_root']
        self.raw_dataset_uri = join(rv_root, 'rv-output', 'raw-datasets',
                                    raw_dataset_key)
        self.dataset_uri = join(self.raw_dataset_uri, 'datasets', dataset_key)
        self.model_uri = join(self.dataset_uri, 'models', model_key)
        self.prediction_uri = join(
            self.model_uri, 'predictions', prediction_key)
        self.eval_uri = join(self.prediction_uri, 'evals', eval_key)

        self.compute_raster_stats_config_uri = self.get_config_uri(
            self.raw_dataset_uri)
        self.make_training_chips_config_uri = self.get_config_uri(
            self.dataset_uri)
        self.train_config_uri = self.get_config_uri(self.model_uri)
        self.predict_config_uri = self.get_config_uri(self.prediction_uri)
        self.eval_config_uri = self.get_config_uri(self.eval_uri)

        self.compute_raster_stats_output_uri = self.get_output_uri(
            self.raw_dataset_uri)
        self.make_training_chips_output_uri = self.get_output_uri(
            self.dataset_uri)
        self.train_output_uri = self.get_output_uri(self.model_uri)
        self.prediction_output_uri = self.get_output_uri(self.prediction_uri)
        self.eval_output_uri = self.get_output_uri(self.eval_uri)

    def get_config_uri(self, prefix_uri):
        return join(prefix_uri, 'config.json')

    def get_output_uri(self, prefix_uri):
        return join(prefix_uri, 'output')


def is_uri_valid(uri):
    parsed_uri = urlparse(uri)

    if parsed_uri.scheme == 's3':
        try:
            s3.Object(parsed_uri.netloc, parsed_uri.path[1:]).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print('URI cannot be found: {}'.format(uri))
                print(e)
                return False
    else:
        if not isfile(uri):
            print('URI cannot be found: {}'.format(uri))
            return False

    return True


def is_validated_uri_field(message_type, field_name):
    return (message_type, field_name) in validated_uri_fields


def is_config_valid(config):
    # If config is primitive, do nothing.
    if not hasattr(config, 'ListFields'):
        return True

    message_type = config.DESCRIPTOR.full_name

    is_valid = True
    for field_desc, field_val in config.ListFields():
        field_name = field_desc.name

        if is_validated_uri_field(message_type, field_name):
            if field_name.endswith('uri'):
                is_valid = is_uri_valid(field_val) and is_valid

            if field_name.endswith('uris'):
                for uri in field_val:
                    is_valid = is_uri_valid(uri) and is_valid

        # Recurse.
        if field_desc.label == FieldDescriptor.LABEL_REPEATED:
            for field_val_item in field_val:
                is_valid = \
                    is_config_valid(field_val_item) and is_valid
        else:
            is_valid = is_config_valid(field_val) and is_valid

    return is_valid


def apply_uri_map(config, uri_map):
    """Do parameter substitution on any URI fields."""
    def _apply_uri_map(config):
        # If config is primitive, do nothing.
        if not hasattr(config, 'ListFields'):
            return

        # For each field in message, update its value if the name ends with
        # uri or uris.
        for field_desc, field_val in config.ListFields():
            field_name = field_desc.name

            if field_name.endswith('uri'):
                new_uri = field_val.format(**uri_map)
                setattr(config, field_name, new_uri)

            if field_name.endswith('uris'):
                for ind, uri in enumerate(field_val):
                    new_uri = uri.format(**uri_map)
                    field_val[ind] = new_uri

            # Recurse.
            if field_desc.label == FieldDescriptor.LABEL_REPEATED:
                for field_val_item in field_val:
                    _apply_uri_map(field_val_item)
            else:
                _apply_uri_map(field_val)

    new_config = config.__deepcopy__()
    _apply_uri_map(new_config)
    return new_config


class ChainWorkflow(object):
    def __init__(self, workflow_uri, remote=False):
        self.workflow = load_json_config(workflow_uri, ChainWorkflowConfig())
        self.uri_map = (self.workflow.remote_uri_map
                        if remote else self.workflow.local_uri_map)
        is_valid = is_config_valid(apply_uri_map(self.workflow, self.uri_map))
        if not is_valid:
            exit()

        self.path_generator = PathGenerator(
            self.uri_map, self.workflow.raw_dataset_key,
            self.workflow.dataset_key, self.workflow.model_key,
            self.workflow.prediction_key, self.workflow.eval_key)

        self.update_raster_transformer()
        self.update_scenes()

    def update_raster_transformer(self):
        stats_uri = join(
            self.path_generator.compute_raster_stats_output_uri, 'stats.json')
        self.workflow.raster_transformer.stats_uri = stats_uri

    def update_scenes(self):
        for idx, scene in enumerate(self.workflow.train_scenes):
            if len(scene.id) < 1:
                scene.id = 'train-{}'.format(idx)
            # Set raster_tranformer for raster_sources
            scene.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

        for idx, scene in enumerate(self.workflow.test_scenes):
            if len(scene.id) < 1:
                scene.id = 'eval-{}'.format(idx)
            scene.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

            # Set prediction_label_store from generated URI.
            scene.prediction_label_store.MergeFrom(
                self.make_prediction_label_store(scene))

        for idx, scene in enumerate(self.workflow.predict_scenes):
            if len(scene.id) < 1:
                scene.id = 'predict-{}'.format(idx)
            scene.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

            # Set prediction_label_store from generated URI.
            scene.prediction_label_store.MergeFrom(
                self.make_prediction_label_store(scene))

    def make_prediction_label_store(self, scene):
        label_store = scene.ground_truth_label_store
        label_store_type = label_store.WhichOneof(
            'label_store_type')
        prediction_uri = join(
            self.path_generator.prediction_output_uri,
            '{}.json'.format(scene.id))

        if label_store_type == 'object_detection_geojson_file':
            geojson_file = ObjectDetectionGeoJSONFileConfig(uri=prediction_uri)
            return LabelStoreConfig(
                object_detection_geojson_file=geojson_file)
        elif label_store_type == 'classification_geojson_file':
            geojson_file = ClassificationGeoJSONFileConfig(uri=prediction_uri)
            return LabelStoreConfig(
                classification_geojson_file=geojson_file)
        else:
            raise ValueError(
                'Not sure how to generate label source config for type {}'
                .format(label_store_type))

    def get_compute_raster_stats_config(self):
        config = ComputeRasterStatsConfig()
        scenes = copy.deepcopy(self.workflow.train_scenes)
        scenes.extend(self.workflow.test_scenes)
        scenes.extend(self.workflow.predict_scenes)
        for scene in scenes:
            # Set the raster_transformer so its fields are null since
            # compute_raster_stats will generate stats_uri.
            raster_source = copy.deepcopy(scene.raster_source)
            raster_source.raster_transformer.stats_uri = ''
            config.raster_sources.extend([raster_source])
        config.stats_uri = self.workflow.raster_transformer.stats_uri
        config = apply_uri_map(config, self.uri_map)
        return config

    def get_make_training_chips_config(self):
        config = MakeTrainingChipsConfig()
        config.train_scenes.MergeFrom(self.workflow.train_scenes)
        config.validation_scenes.MergeFrom(self.workflow.test_scenes)
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.make_training_chips_options)
        config.options.chip_size = self.workflow.chip_size
        config.options.debug = self.workflow.debug
        config.options.output_uri = \
            self.path_generator.make_training_chips_output_uri

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_train_config(self):
        config = TrainConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.train_options)
        config.options.training_data_uri = \
            self.path_generator.make_training_chips_output_uri
        config.options.output_uri = \
            self.path_generator.train_output_uri

        # Copy backend config so that it is nested under model_uri. This way,
        # all config files and corresponding output of RV will be located next
        # to each other in the file system.
        backend_config_copy_uri = join(
            self.path_generator.model_uri, 'backend.config')
        backend_config_uri = config.options.backend_config_uri.format(
            **self.uri_map)
        backend_config_str = file_to_str(backend_config_uri)
        str_to_file(backend_config_str, backend_config_copy_uri)
        config.options.backend_config_uri = backend_config_copy_uri

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_predict_config(self):
        config = PredictConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.scenes.MergeFrom(self.workflow.test_scenes)
        config.scenes.MergeFrom(self.workflow.predict_scenes)
        config.options.MergeFrom(self.workflow.predict_options)
        config.options.debug = self.workflow.debug
        config.options.debug_uri = join(
            self.path_generator.prediction_output_uri, 'debug')
        config.options.chip_size = self.workflow.chip_size
        config.options.model_uri = join(
            self.path_generator.train_output_uri, 'model')
        config = apply_uri_map(config, self.uri_map)
        return config

    def get_eval_config(self):
        config = EvalConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.scenes.MergeFrom(self.workflow.test_scenes)
        config.options.MergeFrom(self.workflow.eval_options)
        config.options.debug = self.workflow.debug
        config.options.output_uri = join(
            self.path_generator.eval_output_uri, 'eval.json')
        config = apply_uri_map(config, self.uri_map)
        return config

    def save_configs(self, tasks):
        print('Generating and saving config files...')

        if COMPUTE_RASTER_STATS in tasks:
            save_json_config(
                self.get_compute_raster_stats_config(),
                self.path_generator.compute_raster_stats_config_uri)

        if MAKE_TRAINING_CHIPS in tasks:
            save_json_config(
                self.get_make_training_chips_config(),
                self.path_generator.make_training_chips_config_uri)

        if TRAIN in tasks:
            save_json_config(self.get_train_config(),
                             self.path_generator.train_config_uri)

        if PREDICT in tasks:
            save_json_config(self.get_predict_config(),
                             self.path_generator.predict_config_uri)

        if EVAL in tasks:
            save_json_config(self.get_eval_config(),
                             self.path_generator.eval_config_uri)

    def remote_run(self, tasks, branch):
        # Run everything in GPU queue since Batch doesn't seem to
        # handle dependencies across different queues.
        parent_job_ids = []

        if COMPUTE_RASTER_STATS in tasks:
            command = make_command(
                COMPUTE_RASTER_STATS,
                self.path_generator.compute_raster_stats_config_uri)
            job_id = _batch_submit(branch, command, attempts=1, gpu=True)
            parent_job_ids = [job_id]

        if MAKE_TRAINING_CHIPS in tasks:
            command = make_command(
                MAKE_TRAINING_CHIPS,
                self.path_generator.make_training_chips_config_uri)
            job_id = _batch_submit(branch, command, attempts=1, gpu=True,
                                   parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if TRAIN in tasks:
            command = make_command(
                TRAIN, self.path_generator.train_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if PREDICT in tasks:
            command = make_command(
                PREDICT, self.path_generator.predict_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if EVAL in tasks:
            command = make_command(
                EVAL, self.path_generator.eval_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=True,
                parent_job_ids=parent_job_ids)

    def local_run(self, tasks):
        if COMPUTE_RASTER_STATS in tasks:
            run._compute_raster_stats(
                self.path_generator.compute_raster_stats_config_uri)

        if MAKE_TRAINING_CHIPS in tasks:
            run._make_training_chips(
                self.path_generator.make_training_chips_config_uri)

        if TRAIN in tasks:
            run._train(self.path_generator.train_config_uri)

        if PREDICT in tasks:
            run._predict(self.path_generator.predict_config_uri)

        if EVAL in tasks:
            run._eval(self.path_generator.eval_config_uri)


@click.command()
@click.argument('workflow_uri')
@click.argument('tasks', nargs=-1)
@click.option('--remote', is_flag=True)
@click.option('--simulated-remote', is_flag=True)
@click.option('--branch', default='develop')
@click.option('--run', is_flag=True)
def main(workflow_uri, tasks, remote, simulated_remote, branch, run):
    if len(tasks) == 0:
        tasks = ALL_TASKS

    for task in tasks:
        if task not in ALL_TASKS:
            raise Exception("Task '{}' is not a valid task.".format(task))

    workflow = ChainWorkflow(workflow_uri, remote=(remote or simulated_remote))
    workflow.save_configs(tasks)

    if run:
        if remote:
            workflow.remote_run(tasks, branch)
        else:
            workflow.local_run(tasks)


if __name__ == '__main__':
    main()
