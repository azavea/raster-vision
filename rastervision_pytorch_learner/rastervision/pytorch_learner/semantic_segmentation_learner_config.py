from typing import Callable, Optional
from os.path import join
from enum import Enum
import logging

import albumentations as A
from torch import nn
from torch.utils.data import Dataset
from torchvision import models

from rastervision.core.data import Scene
from rastervision.core.rv_pipeline import WindowSamplingMethod
from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator, ConfigError)
from rastervision.pytorch_learner.learner_config import (
    Backbone, LearnerConfig, ModelConfig, ImageDataConfig, GeoDataConfig)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationImageDataset,
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationRandomWindowGeoDataset)
from rastervision.pytorch_learner.utils import adjust_conv_channels

log = logging.getLogger(__name__)


class SemanticSegmentationDataFormat(Enum):
    default = 'default'


def ss_data_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 2:
        cfg_dict['type_hint'] = 'semantic_segmentation_image_data'
    elif version < 3:
        try:
            # removed in version 3
            del cfg_dict['channel_display_groups']
        except KeyError:
            pass
    return cfg_dict


def ss_image_data_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 3:
        try:
            # removed in version 3
            del cfg_dict['img_format']
            del cfg_dict['label_format']
            del cfg_dict['channel_display_groups']
        except KeyError:
            pass
    return cfg_dict


@register_config(
    'semantic_segmentation_data', upgrader=ss_data_config_upgrader)
class SemanticSegmentationDataConfig(Config):
    pass


@register_config(
    'semantic_segmentation_image_data', upgrader=ss_image_data_config_upgrader)
class SemanticSegmentationImageDataConfig(SemanticSegmentationDataConfig,
                                          ImageDataConfig):
    """Configure :class:`SemanticSegmentationImageDatasets <.SemanticSegmentationImageDataset>`.

    This assumes the following file structure:

    .. code-block:: text

        <data_dir>/
            img/
                <img 1>.<extension>
                <img 2>.<extension>
                ...
                <img N>.<extension>
            labels/
                <img 1>.<extension>
                <img 2>.<extension>
                ...
                <img N>.<extension>

    """ # noqa
    data_format: SemanticSegmentationDataFormat = (
        SemanticSegmentationDataFormat.default)

    def update(self, *args, **kwargs):
        SemanticSegmentationDataConfig.update(self)
        ImageDataConfig.update(self, *args, **kwargs)

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        if self.data_format != SemanticSegmentationDataFormat.default:
            raise NotImplementedError()
        img_dir = join(data_dir, 'img')
        label_dir = join(data_dir, 'labels')
        ds = SemanticSegmentationImageDataset(
            img_dir=img_dir, label_dir=label_dir, transform=transform)
        return ds


@register_config('semantic_segmentation_geo_data')
class SemanticSegmentationGeoDataConfig(SemanticSegmentationDataConfig,
                                        GeoDataConfig):
    """Configure semantic segmentation :class:`GeoDatasets <.GeoDataset>`.

    See
    :mod:`rastervision.pytorch_learner.dataset.semantic_segmentation_dataset`.
    """

    def update(self, *args, **kwargs):
        SemanticSegmentationDataConfig.update(self)
        GeoDataConfig.update(self, *args, **kwargs)

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None,
                         for_chipping: bool = False) -> Dataset:
        if isinstance(self.sampling, dict):
            opts = self.sampling[scene.id]
        else:
            opts = self.sampling

        extra_args = {}
        if for_chipping:
            extra_args = dict(
                normalize=False, to_pytorch=False, return_window=True)

        if opts.method == WindowSamplingMethod.sliding:
            ds = SemanticSegmentationSlidingWindowGeoDataset(
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
            ds = SemanticSegmentationRandomWindowGeoDataset(
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


@register_config('semantic_segmentation_model')
class SemanticSegmentationModelConfig(ModelConfig):
    """Configure a semantic segmentation model."""

    backbone: Backbone = Field(
        Backbone.resnet50,
        description='The torchvision.models backbone to use. Currently, only '
        'resnet50 and resnet101 are supported.')

    @validator('backbone')
    def only_valid_backbones(cls, v):
        if v not in [Backbone.resnet50, Backbone.resnet101]:
            raise ValueError(
                'The only valid backbones for DeepLabv3 are resnet50 '
                'and resnet101.')
        return v

    def build_default_model(self, num_classes: int,
                            in_channels: int) -> nn.Module:
        backbone_name = self.get_backbone_str()
        pretrained = self.pretrained
        weights = 'DEFAULT' if pretrained else None
        model_factory_func: Callable = getattr(models.segmentation,
                                               f'deeplabv3_{backbone_name}')
        model = model_factory_func(
            num_classes=num_classes,
            weights_backbone=weights,
            aux_loss=False,
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
                    'into Raster Vision. See isprs_potsdam.py for an example '
                    'of how to import external models. Alternatively, you can '
                    'override this function.')
            model.backbone.conv1 = adjust_conv_channels(
                old_conv=model.backbone.conv1,
                in_channels=in_channels,
                pretrained=pretrained)
        return model


@register_config('semantic_segmentation_learner')
class SemanticSegmentationLearnerConfig(LearnerConfig):
    """Configure a :class:`.SemanticSegmentationLearner`."""

    model: Optional[SemanticSegmentationModelConfig]

    def build(self,
              tmp_dir=None,
              model_weights_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.semantic_segmentation_learner import (
            SemanticSegmentationLearner)
        return SemanticSegmentationLearner(
            self,
            tmp_dir=tmp_dir,
            model_weights_path=model_weights_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)
