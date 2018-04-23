from os.path import join
import copy

import click
from google.protobuf.descriptor import FieldDescriptor

from rastervision.protos.chain_workflow_pb2 import ChainWorkflowConfig
from rastervision.protos.compute_raster_stats_pb2 import (
    ComputeRasterStatsConfig)
from rastervision.protos.process_training_data_pb2 import (
    ProcessTrainingDataConfig)
from rastervision.protos.train_pb2 import TrainConfig
from rastervision.protos.predict_pb2 import PredictConfig
from rastervision.protos.eval_pb2 import EvalConfig
from rastervision.protos.label_store_pb2 import (
    LabelStore as LabelStoreConfig,
    ObjectDetectionGeoJSONFile as ObjectDetectionGeoJSONFileConfig,
    ClassificationGeoJSONFile as ClassificationGeoJSONFileConfig)
from rastervision.protos.raster_transformer_pb2 import (
    RasterTransformer as RasterTransformerConfig)
from rastervision.protos.raster_source_pb2 import (
    RasterSource as RasterSourceConfig)

from rastervision.utils.files import (
    load_json_config, save_json_config, file_to_str, str_to_file)
from rastervision.utils.batch import _batch_submit
from rastervision import run

COMPUTE_RASTER_STATS = 'compute_raster_stats'
PROCESS_TRAINING_DATA = 'process_training_data'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_TASKS = [COMPUTE_RASTER_STATS, PROCESS_TRAINING_DATA, TRAIN, PREDICT, EVAL]


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
        self.process_training_data_config_uri = self.get_config_uri(
            self.dataset_uri)
        self.train_config_uri = self.get_config_uri(self.model_uri)
        self.predict_config_uri = self.get_config_uri(self.prediction_uri)
        self.eval_config_uri = self.get_config_uri(self.eval_uri)

        self.compute_raster_stats_output_uri = self.get_output_uri(
            self.raw_dataset_uri)
        self.process_training_data_output_uri = self.get_output_uri(
            self.dataset_uri)
        self.train_output_uri = self.get_output_uri(self.model_uri)
        self.prediction_output_uri = self.get_output_uri(self.prediction_uri)
        self.eval_output_uri = self.get_output_uri(self.eval_uri)

    def get_config_uri(self, prefix_uri):
        return join(prefix_uri, 'config.json')

    def get_output_uri(self, prefix_uri):
        return join(prefix_uri, 'output')


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
                setattr(config, field_name, field_val.format(**uri_map))

            if field_name.endswith('uris'):
                for ind, uri in enumerate(field_val):
                    field_val[ind] = uri.format(**uri_map)
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
        self.path_generator = PathGenerator(
            self.uri_map, self.workflow.raw_dataset_key,
            self.workflow.dataset_key, self.workflow.model_key,
            self.workflow.prediction_key, self.workflow.eval_key)

        self.update_raster_transformer()
        self.update_projects()

    def update_raster_transformer(self):
        stats_uri = join(
            self.path_generator.compute_raster_stats_output_uri, 'stats.json')
        self.workflow.raster_transformer.stats_uri = stats_uri

    def update_projects(self):
        for idx, project in enumerate(self.workflow.train_projects):
            if len(project.id) < 1:
                project.id = 'train-{}'.format(idx)
            # Set raster_tranformer for raster_sources
            project.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

        for idx, project in enumerate(self.workflow.test_projects):
            if len(project.id) < 1:
                project.id = 'eval-{}'.format(idx)
            project.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

            # Set prediction_label_store from generated URI.
            project.prediction_label_store.MergeFrom(
                self.make_prediction_label_store(project))

    def make_prediction_label_store(self, project):
        label_store = project.ground_truth_label_store
        label_store_type = label_store.WhichOneof(
            'label_store_type')
        prediction_uri = join(
            self.path_generator.prediction_output_uri,
            '{}.json'.format(project.id))

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
        projects = copy.deepcopy(self.workflow.train_projects)
        projects.extend(self.workflow.test_projects)
        for project in projects:
            # Set the raster_transformer so its fields are null since
            # compute_raster_stats will generate stats_uri.
            raster_source = copy.deepcopy(project.raster_source)
            raster_source.raster_transformer.stats_uri = ''
            config.raster_sources.extend([raster_source])
        config.stats_uri = self.workflow.raster_transformer.stats_uri
        config = apply_uri_map(config, self.uri_map)
        return config

    def get_process_training_data_config(self):
        config = ProcessTrainingDataConfig()
        config.train_projects.MergeFrom(self.workflow.train_projects)
        config.validation_projects.MergeFrom(self.workflow.test_projects)
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.process_training_data_options)
        config.options.chip_size = self.workflow.chip_size
        config.options.debug = self.workflow.debug
        config.options.output_uri = \
            self.path_generator.process_training_data_output_uri

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_train_config(self):
        config = TrainConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.train_options)
        config.options.training_data_uri = \
            self.path_generator.process_training_data_output_uri
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
        config.projects.MergeFrom(self.workflow.test_projects)
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
        config.projects.MergeFrom(self.workflow.test_projects)
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

        if PROCESS_TRAINING_DATA in tasks:
            save_json_config(
                self.get_process_training_data_config(),
                self.path_generator.process_training_data_config_uri)

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

        if PROCESS_TRAINING_DATA in tasks:
            command = make_command(
                PROCESS_TRAINING_DATA,
                self.path_generator.process_training_data_config_uri)
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

        if PROCESS_TRAINING_DATA in tasks:
            run._process_training_data(
                self.path_generator.process_training_data_config_uri)

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
        if not task in ALL_TASKS:
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
