from rastervision.builders import (
    scene_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.eval import Eval
from rastervision.protos.eval_pb2 import EvalConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, EvalConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    class_map = ml_task.get_class_map()
    scenes = [scene_builder.build(scene_config, class_map)
              for scene_config in config.scenes]
    options = config.options

    return Eval(scenes, ml_task, options)
