from rastervision.builders import (
    project_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.eval import Eval
from rastervision.protos.eval_pb2 import EvalConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, EvalConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    projects = [project_builder.build(project_config)
                for project_config in config.projects]
    options = config.options

    return Eval(projects, ml_task, options)
