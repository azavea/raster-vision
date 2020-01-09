from rastervision.v2.core.config import register_config
from rastervision.v2.learner.learner_config import LearnerConfig, DataConfig


@register_config('classification_data')
class ClassificationDataConfig(DataConfig):
    data_format: str = 'image_folder'


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: ClassificationDataConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision.v2.learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(self, tmp_dir, model_path=model_path)

