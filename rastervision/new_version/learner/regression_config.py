from typing import List

from rastervision.new_version.pipeline.config import register_config
from rastervision.new_version.learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)


@register_config('regression_model')
class RegressionModelConfig(ModelConfig):
    output_multiplier: List[float] = None


@register_config('regression_data')
class RegressionDataConfig(DataConfig):
    pos_labels: List[str] = []
    data_format: str = 'csv'


@register_config('regression_learner')
class RegressionLearnerConfig(LearnerConfig):
    model: RegressionModelConfig
    data: RegressionDataConfig

    def update(self, parent=None):
        super().update(parent)

        if self.model.output_multiplier is None:
            self.model.output_multiplier = [1.0] * len(self.data.labels)

    def get_learner(self):
        from rastervision.new_version.learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner
