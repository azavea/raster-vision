from rv2.builders import (
    project_builder, ml_task_builder)
from rv2.utils import files
from rv2.commands.predict import Predict
from rv2.protos.predict_pb2 import PredictConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, PredictConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    projects = [project_builder.build(project_config)
                for project_config in config.projects]
    options = config.options

    return Predict(projects, ml_task, options)
