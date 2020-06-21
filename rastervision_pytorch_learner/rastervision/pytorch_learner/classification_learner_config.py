from enum import Enum

from rastervision.pipeline.config import register_config
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

    def build(self, tmp_dir, model_path=None):
        from rastervision.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(self, tmp_dir, model_path=model_path)
