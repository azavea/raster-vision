from rastervision.builders import (scene_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.make_chips import MakeChips
from rastervision.protos.make_chips_pb2 import MakeChipsConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, MakeChipsConfig())

    ml_task = ml_task_builder.build(config.model_config)
    class_map = ml_task.get_class_map()
    train_scenes = [
        scene_builder.build(scene_config, class_map)
        for scene_config in config.train_scenes
    ]
    validation_scenes = [
        scene_builder.build(scene_config, class_map)
        for scene_config in config.validation_scenes
    ]
    options = config.options

    return MakeChips(train_scenes, validation_scenes, ml_task, options)
