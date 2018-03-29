from rastervision.builders import (
    project_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.process_training_data import ProcessTrainingData
from rastervision.protos.process_training_data_pb2 import ProcessTrainingDataConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, ProcessTrainingDataConfig())

    train_projects = [project_builder.build(project_config)
                      for project_config in config.train_projects]
    validation_projects = [project_builder.build(project_config)
                           for project_config in config.validation_projects]
    ml_task = ml_task_builder.build(config.machine_learning)
    options = config.options

    return ProcessTrainingData(train_projects, validation_projects, ml_task,
                               options)
