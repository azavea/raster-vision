import click

from rastervision.backend.keras_classification.utils \
    import load_json_config
from rastervision.backend.keras_classification.builders \
    import trainer_builder, model_builder
from rastervision.protos.keras_classification.pipeline_pb2 \
    import PipelineConfig


def _train(config_path, pretrained_model_path, do_monitoring):
    config = load_json_config(config_path, PipelineConfig())
    model = model_builder.build(config.model, pretrained_model_path)
    trainer = trainer_builder.build(config.trainer, model)
    trainer.train(do_monitoring)


@click.command()
@click.argument('config_path')
def train(config_path):
    _train(config_path)


if __name__ == '__main__':
    train()
