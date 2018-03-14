from os.path import join

import click
from google.protobuf.descriptor import FieldDescriptor

from rv2.protos.chain_workflow_pb2 import ChainWorkflowConfig
from rv2.protos.make_train_data_pb2 import MakeTrainDataConfig
from rv2.protos.train_pb2 import TrainConfig
from rv2.protos.predict_pb2 import PredictConfig
from rv2.protos.eval_pb2 import EvalConfig
from rv2.protos.annotation_source_pb2 import (
    AnnotationSource as AnnotationSourceConfig,
    GeoJSONFile as GeoJSONFileConfig)

from rv2.utils.files import (
    load_json_config, save_json_config, file_to_str, str_to_file)
from rv2.utils.batch import _batch_submit
from rv2 import run

MAKE_TRAIN_DATA = 'make_train_data'
TRAIN = 'train'
PREDICT = 'predict'
EVAL = 'eval'
ALL_TASKS = [MAKE_TRAIN_DATA, TRAIN, PREDICT, EVAL]


def make_command(command, config_uri):
    return 'python -m rv2.run {} {}'.format(command, config_uri)


class PathGenerator(object):
    def __init__(self, uri_map, dataset_key, model_key, prediction_key,
                 eval_key):
        rv_root = uri_map['rv_root']
        self.dataset_uri = join(rv_root, 'rv-output', 'datasets', dataset_key)
        self.model_uri = join(self.dataset_uri, 'models', model_key)
        self.prediction_uri = join(
            self.model_uri, 'predictions', prediction_key)
        self.eval_uri = join(self.prediction_uri, 'evals', eval_key)

        self.make_train_data_config_uri = self.get_config_uri(self.dataset_uri)
        self.train_config_uri = self.get_config_uri(self.model_uri)
        self.predict_config_uri = self.get_config_uri(self.prediction_uri)
        self.eval_config_uri = self.get_config_uri(self.eval_uri)

        self.make_train_data_output_uri = self.get_output_uri(self.dataset_uri)
        self.train_output_uri = self.get_output_uri(self.model_uri)
        self.prediction_output_uri = self.get_output_uri(self.prediction_uri)
        self.eval_output_uri = self.get_output_uri(self.eval_uri)

    def get_config_uri(self, prefix_uri):
        return join(prefix_uri, 'config.json')

    def get_output_uri(self, prefix_uri):
        return join(prefix_uri, 'output')


def apply_uri_map(config, uri_map):
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
            self.uri_map, self.workflow.dataset_key, self.workflow.model_key,
            self.workflow.prediction_key, self.workflow.eval_key)

        self.update_projects()

    def update_projects(self):
        for project in self.workflow.train_projects:
            # Set raster_tranformer for raster_sources
            project.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

        for project in self.workflow.test_projects:
            project.raster_source.raster_transformer.MergeFrom(
                self.workflow.raster_transformer)

            # Set prediction_annotation_source from generated URI.
            project.prediction_annotation_source.MergeFrom(
                self.make_prediction_annotation_source(project))

    def make_prediction_annotation_source(self, project):
        annotation_source = project.ground_truth_annotation_source
        annotation_source_type = annotation_source.WhichOneof(
            'annotation_source_type')
        if annotation_source_type == 'geojson_file':
            prediction_uri = join(
                self.path_generator.prediction_output_uri,
                '{}.json'.format(project.id))
            geojson_file = GeoJSONFileConfig(uri=prediction_uri)
            return AnnotationSourceConfig(geojson_file=geojson_file)
        else:
            raise ValueError(
                'Not sure how to generate annotation source config for type {}'
                .format(annotation_source_type))

    def get_make_train_data_config(self):
        config = MakeTrainDataConfig()
        config.train_projects.MergeFrom(self.workflow.train_projects)
        config.validation_projects.MergeFrom(self.workflow.test_projects)
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.make_train_data_options)
        config.options.chip_size = self.workflow.chip_size
        config.options.debug = self.workflow.debug
        config.options.output_uri = \
            self.path_generator.make_train_data_output_uri
        config.label_items.MergeFrom(self.workflow.label_items)

        config = apply_uri_map(config, self.uri_map)
        return config

    def get_train_config(self):
        config = TrainConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.options.MergeFrom(self.workflow.train_options)
        config.options.train_data_uri = \
            self.path_generator.make_train_data_output_uri
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
        config.label_items.MergeFrom(self.workflow.label_items)
        config.projects.MergeFrom(self.workflow.test_projects)
        config.options.MergeFrom(self.workflow.predict_options)
        config.options.debug = self.workflow.debug
        config.options.chip_size = self.workflow.chip_size
        config.options.model_uri = join(
            self.path_generator.train_output_uri, 'model')
        config = apply_uri_map(config, self.uri_map)
        return config

    def get_eval_config(self):
        config = EvalConfig()
        config.machine_learning.MergeFrom(self.workflow.machine_learning)
        config.label_items.MergeFrom(self.workflow.label_items)
        config.projects.MergeFrom(self.workflow.test_projects)
        config.options.MergeFrom(self.workflow.eval_options)
        config.options.debug = self.workflow.debug
        config.options.output_uri = join(
            self.path_generator.eval_output_uri, 'eval.json')
        config = apply_uri_map(config, self.uri_map)
        return config

    def save_configs(self, tasks):
        print('Generating and saving config files...')
        if MAKE_TRAIN_DATA in tasks:
            save_json_config(self.get_make_train_data_config(),
                             self.path_generator.make_train_data_config_uri)
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
        parent_job_ids = []
        if MAKE_TRAIN_DATA in tasks:
            command = make_command(
                'make_train_data',
                self.path_generator.make_train_data_config_uri)
            job_id = _batch_submit(branch, command, attempts=1, gpu=False)
            parent_job_ids = [job_id]

        if TRAIN in tasks:
            command = make_command(
                'train', self.path_generator.train_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=True,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if PREDICT in tasks:
            command = make_command(
                'predict', self.path_generator.predict_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=False,
                parent_job_ids=parent_job_ids)
            parent_job_ids = [job_id]

        if EVAL in tasks:
            command = make_command(
                'eval', self.path_generator.eval_config_uri)
            job_id = _batch_submit(
                branch, command, attempts=1, gpu=False,
                parent_job_ids=parent_job_ids)

    def local_run(self, tasks):
        if MAKE_TRAIN_DATA in tasks:
            run._make_train_data(
                self.path_generator.make_train_data_config_uri)

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
@click.option('--branch', default='develop')
@click.option('--run', is_flag=True)
def main(workflow_uri, tasks, remote, branch, run):
    if len(tasks) == 0:
        tasks = ALL_TASKS

    workflow = ChainWorkflow(workflow_uri, remote=remote)
    workflow.save_configs(tasks)

    if run:
        if remote:
            workflow.remote_run(tasks, branch)
        else:
            workflow.local_run(tasks)


if __name__ == '__main__':
    main()
