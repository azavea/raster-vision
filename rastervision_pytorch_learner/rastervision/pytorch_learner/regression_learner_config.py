from typing import TYPE_CHECKING, Callable, Iterable, List, Optional, Sequence, Union
from enum import Enum
import logging

import albumentations as A
from torch import nn
from torchvision import models

from rastervision.core.data import Scene
from rastervision.core.rv_pipeline import WindowSamplingMethod
from rastervision.pipeline.config import (Config, register_config, Field,
                                          ConfigError)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, PlotOptions, ImageDataConfig, GeoDataConfig)
from rastervision.pytorch_learner.dataset import (
    RegressionImageDataset, RegressionSlidingWindowGeoDataset,
    RegressionRandomWindowGeoDataset)
from rastervision.pytorch_learner.utils import adjust_conv_channels

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


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
    """Configure :class:`RegressionImageDatasets <.RegressionImageDataset>`."""

    data_format: RegressionDataFormat = RegressionDataFormat.csv
    plot_options: Optional[RegressionPlotOptions] = Field(
        RegressionPlotOptions(), description='Options to control plotting.')

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> RegressionImageDataset:
        ds = RegressionImageDataset(
            data_dir, self.class_names, transform=transform)
        return ds


@register_config('regression_geo_data')
class RegressionGeoDataConfig(RegressionDataConfig, GeoDataConfig):
    """Configure regression :class:`GeoDatasets <.GeoDataset>`.

    See :mod:`rastervision.pytorch_learner.dataset.regression_dataset`.
    """

    plot_options: Optional[RegressionPlotOptions] = Field(
        RegressionPlotOptions(), description='Options to control plotting.')

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None,
                         for_chipping: bool = False
                         ) -> Union[RegressionSlidingWindowGeoDataset,
                                    RegressionRandomWindowGeoDataset]:
        if isinstance(self.sampling, dict):
            opts = self.sampling[scene.id]
        else:
            opts = self.sampling

        extra_args = {}
        if for_chipping:
            extra_args = dict(
                normalize=False, to_pytorch=False, return_window=True)

        if opts.method == WindowSamplingMethod.sliding:
            ds = RegressionSlidingWindowGeoDataset(
                scene,
                size=opts.size,
                stride=opts.stride,
                padding=opts.padding,
                pad_direction=opts.pad_direction,
                within_aoi=opts.within_aoi,
                transform=transform,
                **extra_args,
            )
        elif opts.method == WindowSamplingMethod.random:
            ds = RegressionRandomWindowGeoDataset(
                scene,
                size_lims=opts.size_lims,
                h_lims=opts.h_lims,
                w_lims=opts.w_lims,
                out_size=opts.size,
                padding=opts.padding,
                max_windows=opts.max_windows,
                max_sample_attempts=opts.max_sample_attempts,
                efficient_aoi_sampling=opts.efficient_aoi_sampling,
                within_aoi=opts.within_aoi,
                transform=transform,
                **extra_args,
            )
        else:
            raise NotImplementedError()
        return ds


class RegressionModel(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 out_features: int,
                 pos_out_inds: Optional[Sequence[int]] = None,
                 prob_out_inds: Optional[Sequence[int]] = None,
                 **kwargs):
        super().__init__()
        self.backbone = backbone
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, out_features)
        self.pos_out_inds = pos_out_inds
        self.prob_out_inds = prob_out_inds

    def forward(self, x: 'torch.Tensor') -> 'torch.Tensor':
        out: 'torch.Tensor' = self.backbone(x)
        if self.pos_out_inds:
            for ind in self.pos_out_inds:
                out[:, ind] = out[:, ind].exp()
        if self.prob_out_inds:
            for ind in self.prob_out_inds:
                out[:, ind] = out[:, ind].sigmoid()
        return out


@register_config('regression_model')
class RegressionModelConfig(ModelConfig):
    """Configure a regression model."""

    output_multiplier: List[float] = None

    def update(self, learner=None):
        if learner is not None and self.output_multiplier is None:
            self.output_multiplier = [1.0] * len(learner.data.class_names)

    def build_default_model(
            self,
            num_classes: int,
            in_channels: int,
            class_names: Optional[Sequence[str]] = None,
            pos_class_names: Optional[Iterable[str]] = None,
            prob_class_names: Optional[Iterable[str]] = None) -> nn.Module:
        backbone_name = self.get_backbone_str()
        pretrained = self.pretrained
        out_features = num_classes

        pos_out_inds = None
        if pos_class_names is not None:
            pos_out_inds = [
                class_names.index(class_name) for class_name in pos_class_names
            ]
        prob_out_inds = None
        if prob_class_names is not None:
            prob_out_inds = [
                class_names.index(class_name)
                for class_name in prob_class_names
            ]

        weights = 'DEFAULT' if pretrained else None
        model_factory_func: Callable[..., nn.Module] = getattr(
            models, backbone_name)
        backbone = model_factory_func(weights=weights, **self.extra_args)
        model = RegressionModel(
            backbone,
            out_features,
            pos_out_inds=pos_out_inds,
            prob_out_inds=prob_out_inds,
            **self.extra_args)

        if in_channels != 3:
            if not backbone_name.startswith('resnet'):
                raise ConfigError(
                    'All TorchVision backbones do not provide the same API '
                    'for accessing the first conv layer. '
                    'Therefore, conv layer modification to support '
                    'arbitrary input channels is only supported for resnet '
                    'backbones. To use other backbones, it is recommended to '
                    'fork the TorchVision repo, define factory functions or '
                    'subclasses that perform the necessary modifications, and '
                    'then use the external model functionality to import it '
                    'into Raster Vision. See spacenet_rio.py for an example '
                    'of how to import external models. Alternatively, you can '
                    'override this function.')
            model.backbone.conv1 = adjust_conv_channels(
                old_conv=model.backbone.conv1,
                in_channels=in_channels,
                pretrained=pretrained)

        return model


@register_config('regression_learner')
class RegressionLearnerConfig(LearnerConfig):
    """Configure a :class:`.RegressionLearner`."""

    model: Optional[RegressionModelConfig]

    def build(self,
              tmp_dir,
              model_weights_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.regression_learner import (
            RegressionLearner)
        return RegressionLearner(
            self,
            tmp_dir,
            model_weights_path=model_weights_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)
