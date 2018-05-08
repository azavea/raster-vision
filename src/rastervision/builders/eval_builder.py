from rastervision.builders import (
    project_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.eval import Eval
from rastervision.protos.eval_pb2 import EvalConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, EvalConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    class_map = ml_task.get_class_map()
    projects = [project_builder.build(project_config, class_map)
                for project_config in config.projects]
    options = config.options

    return Eval(projects, ml_task, options)
