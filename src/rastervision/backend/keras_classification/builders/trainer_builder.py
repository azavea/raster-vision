from rastervision.backend.keras_classification.builders import optimizer_builder
from rastervision.backend.keras_classification.core.trainer import Trainer


def build(trainer_config, model):
    optimizer = optimizer_builder.build(trainer_config.optimizer)
    trainer = Trainer(model, optimizer, trainer_config.options)
    return trainer
