from typing import List, Optional
from enum import Enum

from rastervision.pipeline.config import register_config, Field
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig, PlotOptions)


class RegressionDataFormat(Enum):
    csv = 'csv'


@register_config('regression_plot_options')
class RegressionPlotOptions(PlotOptions):
    max_scatter_points: int = Field(
        5000,
        description=('Maximum number of datapoints to use in scatter plot. '
                     'Useful to avoid running out of memory and cluttering.'))
    hist_bins: int = Field(
        30, description='Number of bins to use for histogram.')


@register_config('regression_data')
class RegressionDataConfig(DataConfig):
    pos_class_names: List[str] = []
    prob_class_names: List[str] = []
    data_format: RegressionDataFormat = RegressionDataFormat.csv
    plot_options: Optional[RegressionPlotOptions] = Field(
        RegressionPlotOptions(), description='Options to control plotting.')


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

    def build(self,
              tmp_dir,
              model_path=None,
              model_def_path=None,
              loss_def_path=None):
        from rastervision.pytorch_learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner(
            self,
            tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path)
