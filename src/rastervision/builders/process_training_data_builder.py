from rastervision.builders import (
    project_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.process_training_data import ProcessTrainingData
from rastervision.protos.process_training_data_pb2 import ProcessTrainingDataConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, ProcessTrainingDataConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    class_map = ml_task.get_class_map()
    train_projects = [project_builder.build(project_config, class_map)
                      for project_config in config.train_projects]
    validation_projects = [project_builder.build(project_config, class_map)
                           for project_config in config.validation_projects]
    options = config.options

    return ProcessTrainingData(train_projects, validation_projects, ml_task,
                               options)
