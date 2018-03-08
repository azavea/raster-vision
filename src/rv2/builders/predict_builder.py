from rv2.builders import (
    project_builder, ml_method_builder, label_map_builder)
from rv2.utils import files
from rv2.commands.predict import Predict
from rv2.protos.predict_pb2 import PredictConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, PredictConfig())

    ml_method = ml_method_builder.build(config.machine_learning)
    label_map = label_map_builder.build(config.label_items)
    projects = [project_builder.build(project_config)
                for project_config in config.projects]
    options = config.options

    return Predict(projects, ml_method, label_map, options)
