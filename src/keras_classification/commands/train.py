import click

from keras_classification.utils import load_json_config
from keras_classification.builders import trainer_builder, model_builder
from keras_classification.protos.pipeline_pb2 import PipelineConfig


def _train(config_path):
    config = load_json_config(config_path, PipelineConfig())
    model = model_builder.build(config.model)
    trainer = trainer_builder.build(config.trainer, model)
    trainer.train()


@click.command()
@click.argument('config_path')
def train(config_path):
    _train(config_path)


if __name__ == '__main__':
    train()
