from typing import List

from rastervision.new_version.pipeline.config import register_config
from rastervision.new_version.learner.config import LearnerConfig, DataConfig


@register_config('classification_data')
class ClassificationDataConfig(DataConfig):
    data_format: str = 'image_folder'


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: ClassificationDataConfig

    def get_learner(self):
        from rastervision.new_version.learner.classification.learner import ClassificationLearner
        return ClassificationLearner
