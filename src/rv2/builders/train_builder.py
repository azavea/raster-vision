from rv2.builders import ml_task_builder
from rv2.utils import files
from rv2.commands.train import Train
from rv2.protos.train_pb2 import TrainConfig


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, TrainConfig())

    ml_task = ml_task_builder.build(config.machine_learning)
    options = config.options

    return Train(ml_task, options)
