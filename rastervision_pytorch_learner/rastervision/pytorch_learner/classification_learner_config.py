from enum import Enum

from rastervision.pipeline.config import register_config, ConfigError
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)


class ClassificationDataFormat(Enum):
    image_folder = 'image_folder'


@register_config('classification_data')
class ClassificationDataConfig(DataConfig):
    data_format: ClassificationDataFormat = ClassificationDataFormat.image_folder


@register_config('classification_model')
class ClassificationModelConfig(ModelConfig):
    pass


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: ClassificationDataConfig
    model: ClassificationModelConfig

    def build(self, tmp_dir, model_path=None, model_def_path=None):
        from rastervision.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(
            self,
            tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path)

    def validate_config(self):
        super().validate_config()
        self.validate_class_loss_weights()

    def validate_class_loss_weights(self):
        if self.solver.class_loss_weights is None:
            return

        num_weights = len(self.solver.class_loss_weights)
        num_classes = len(self.data.class_names)
        if num_weights != num_classes:
            raise ConfigError(
                f'class_loss_weights ({num_weights}) must be same length as '
                f'the number of classes ({num_classes})')
