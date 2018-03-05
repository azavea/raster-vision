from rv2.builders import (
    project_builder, ml_method_builder, label_map_builder)
from rv2.utils import files
from rv2.commands.eval import Eval
from rv2.protos.eval_pb2 import Eval as EvalPB


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, EvalPB())

    ml_method = ml_method_builder.build(config.machine_learning)
    label_map = label_map_builder.build(config.label_items)
    projects = [project_builder.build(project_config)
                for project_config in config.projects]
    options = config.options

    return Eval(projects, ml_method, label_map, options)
