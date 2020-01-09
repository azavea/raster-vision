from typing import List

from rastervision.v2.core.config import register_config
from rastervision.v2.learner.learner_config import (LearnerConfig, DataConfig,
                                                    ModelConfig)


@register_config('regression_model')
class RegressionModelConfig(ModelConfig):
    output_multiplier: List[float] = None

    def update(self, learner=None):
        if learner is not None and self.output_multiplier is None:
            self.model.output_multiplier = [1.0] * len(
                learner.data.class_names)


@register_config('regression_data')
class RegressionDataConfig(DataConfig):
    pos_labels: List[str] = []
    data_format: str = 'csv'


@register_config('regression_learner')
class RegressionLearnerConfig(LearnerConfig):
    model: RegressionModelConfig
    data: RegressionDataConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision.v2.learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner(self, tmp_dir, model_path=model_path)
