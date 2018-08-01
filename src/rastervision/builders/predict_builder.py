from rastervision.builders import (scene_builder, ml_task_builder)
from rastervision.utils import files
from rastervision.commands.predict import Predict
from rastervision.protos.predict_pb2 import PredictConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, PredictConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    class_map = ml_task.get_class_map()
    scenes = [
        scene_builder.build(scene_config, class_map)
        for scene_config in config.scenes
    ]
    return Predict(scenes, ml_task, config)
