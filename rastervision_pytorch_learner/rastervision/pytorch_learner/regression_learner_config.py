from typing import List
from enum import Enum

from rastervision.pipeline.config import register_config
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)


class RegressionDataFormat(Enum):
    csv = 'csv'


@register_config('regression_data')
class RegressionDataConfig(DataConfig):
    pos_class_names: List[str] = []
    data_format: RegressionDataFormat = RegressionDataFormat.csv


@register_config('regression_model')
class RegressionModelConfig(ModelConfig):
    output_multiplier: List[float] = None

    def update(self, learner=None):
        if learner is not None and self.output_multiplier is None:
            self.output_multiplier = [1.0] * len(learner.data.class_names)


@register_config('regression_learner')
class RegressionLearnerConfig(LearnerConfig):
    model: RegressionModelConfig
    data: RegressionDataConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision.pytorch_learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner(self, tmp_dir, model_path=model_path)
