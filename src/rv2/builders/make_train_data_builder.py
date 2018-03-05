from rv2.builders import (
    project_builder, ml_method_builder, label_map_builder)
from rv2.utils import files
from rv2.commands.make_train_data import MakeTrainData
from rv2.protos.make_train_data_pb2 import MakeTrainData as MakeTrainDataPB


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, MakeTrainDataPB())

    train_projects = [project_builder.build(project_config)
                      for project_config in config.train_projects]
    validation_projects = [project_builder.build(project_config)
                           for project_config in config.validation_projects]
    ml_method = ml_method_builder.build(config.machine_learning)
    label_map = label_map_builder.build(config.label_items)
    options = config.options

    return MakeTrainData(train_projects, validation_projects, ml_method,
                         label_map, options)
