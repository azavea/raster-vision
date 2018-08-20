from rastervision.builders import ml_task_builder
from rastervision.utils import files
from rastervision.commands.train import Train
from rastervision.protos.train_pb2 import TrainConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, TrainConfig())

    ml_task = ml_task_builder.build(config.model_config)
    options = config.options

    return Train(ml_task, options)
