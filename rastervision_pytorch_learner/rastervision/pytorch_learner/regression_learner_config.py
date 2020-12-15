from typing import List, Optional, Union
from enum import Enum

import albumentations as A

from torch.utils.data import Dataset

from rastervision.core.data import Scene
from rastervision.pipeline.config import (Config, register_config, Field)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, PlotOptions, ImageDataConfig, GeoDataConfig,
    GeoDataWindowMethod)
from rastervision.pytorch_learner.dataset import (
    RegressionImageDataset, RegressionSlidingWindowGeoDataset,
    RegressionRandomWindowGeoDataset)


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


def reg_data_config_upgrader(cfg_dict, version):
    if version == 1:
        cfg_dict['type_hint'] = 'regression_image_data'
    return cfg_dict


@register_config('regression_data', upgrader=reg_data_config_upgrader)
class RegressionDataConfig(Config):
    pos_class_names: List[str] = []
    prob_class_names: List[str] = []


@register_config('regression_image_data')
class RegressionImageDataConfig(RegressionDataConfig, ImageDataConfig):
    data_format: RegressionDataFormat = RegressionDataFormat.csv
    plot_options: Optional[RegressionPlotOptions] = Field(
        RegressionPlotOptions(), description='Options to control plotting.')

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        ds = RegressionImageDataset(
            data_dir, self.class_names, transform=transform)
        return ds


@register_config('regression_geo_data')
class RegressionGeoDataConfig(RegressionDataConfig, GeoDataConfig):
    plot_options: Optional[RegressionPlotOptions] = Field(
        RegressionPlotOptions(), description='Options to control plotting.')

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None
                         ) -> Dataset:
        if isinstance(self.window_opts, dict):
            opts = self.window_opts[scene.id]
        else:
            opts = self.window_opts

        if opts.method == GeoDataWindowMethod.sliding:
            ds = RegressionSlidingWindowGeoDataset(
                scene,
                size=opts.size,
                stride=opts.stride,
                padding=opts.padding,
                transform=transform)
        elif opts.method == GeoDataWindowMethod.random:
            ds = RegressionRandomWindowGeoDataset(
                scene,
                size_lims=opts.size_lims,
                h_lims=opts.h_lims,
                w_lims=opts.w_lims,
                out_size=opts.size,
                padding=opts.padding,
                max_windows=opts.max_windows,
                max_sample_attempts=opts.max_sample_attempts,
                transform=transform)
        else:
            raise NotImplementedError()
        return ds


@register_config('regression_model')
class RegressionModelConfig(ModelConfig):
    output_multiplier: List[float] = None

    def update(self, learner=None):
        if learner is not None and self.output_multiplier is None:
            self.output_multiplier = [1.0] * len(learner.data.class_names)


@register_config('regression_learner')
class RegressionLearnerConfig(LearnerConfig):
    model: RegressionModelConfig
    data: Union[RegressionImageDataConfig, RegressionGeoDataConfig]

    def build(self,
              tmp_dir,
              model_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner(
            self,
            tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)
