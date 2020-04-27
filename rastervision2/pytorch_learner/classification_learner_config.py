from enum import Enum

from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)


class DataFormat(Enum):
    image_folder = 1


@register_config('classification_data')
class ClassificationDataConfig(DataConfig):
    data_format: DataFormat = DataFormat.image_folder


@register_config('classification_model')
class ClassificationModelConfig(ModelConfig):
    pass


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: ClassificationDataConfig
    model: ClassificationModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(self, tmp_dir, model_path=model_path)
