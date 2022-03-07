from enum import Enum
from typing import Optional, Union

import albumentations as A
from torch.utils.data import Dataset

from rastervision.core.data import Scene
from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator, ConfigError)
from rastervision.pytorch_learner.learner_config import (
    Backbone, LearnerConfig, ModelConfig, ImageDataConfig, GeoDataConfig,
    GeoDataWindowMethod)
from rastervision.pytorch_learner.dataset import (
    SemanticSegmentationImageDataset,
    SemanticSegmentationSlidingWindowGeoDataset,
    SemanticSegmentationRandomWindowGeoDataset)


class SemanticSegmentationDataFormat(Enum):
    default = 'default'


def ss_data_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 2:
        cfg_dict['type_hint'] = 'semantic_segmentation_image_data'
    elif version < 3:
        try:
            # removed in version 3
            del cfg_dict['img_channels']
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
    data_format: SemanticSegmentationDataFormat = (
        SemanticSegmentationDataFormat.default)

    def update(self, *args, **kwargs):
        SemanticSegmentationDataConfig.update(self)
        ImageDataConfig.update(self, *args, **kwargs)

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        ds = SemanticSegmentationImageDataset(
            data_dir=data_dir, transform=transform)
        return ds


@register_config('semantic_segmentation_geo_data')
class SemanticSegmentationGeoDataConfig(SemanticSegmentationDataConfig,
                                        GeoDataConfig):
    def update(self, *args, **kwargs):
        SemanticSegmentationDataConfig.update(self)
        GeoDataConfig.update(self, *args, **kwargs)

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None
                         ) -> Dataset:
        if isinstance(self.window_opts, dict):
            opts = self.window_opts[scene.id]
        else:
            opts = self.window_opts

        if opts.method == GeoDataWindowMethod.sliding:
            ds = SemanticSegmentationSlidingWindowGeoDataset(
                scene,
                size=opts.size,
                stride=opts.stride,
                padding=opts.padding,
                transform=transform)
        elif opts.method == GeoDataWindowMethod.random:
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
                transform=transform)
        else:
            raise NotImplementedError()
        return ds


@register_config('semantic_segmentation_model')
class SemanticSegmentationModelConfig(ModelConfig):
    backbone: Backbone = Field(
        Backbone.resnet50,
        description=(
            'The torchvision.models backbone to use. At the moment only '
            'resnet50 or resnet101 will work.'))

    @validator('backbone')
    def only_valid_backbones(cls, v):
        if v not in [Backbone.resnet50, Backbone.resnet101]:
            raise ValueError(
                'The only valid backbones for DeepLabv3 are resnet50 '
                'and resnet101.')
        return v


@register_config('semantic_segmentation_learner')
class SemanticSegmentationLearnerConfig(LearnerConfig):
    data: Union[SemanticSegmentationImageDataConfig,
                SemanticSegmentationGeoDataConfig]
    model: SemanticSegmentationModelConfig

    def build(self,
              tmp_dir,
              model_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.semantic_segmentation_learner import (
            SemanticSegmentationLearner)
        return SemanticSegmentationLearner(
            self,
            tmp_dir=tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)

    def validate_config(self):
        super().validate_config()
        self.validate_class_loss_weights()

    def validate_class_loss_weights(self):
        if self.solver.class_loss_weights is None:
            return

        num_weights = len(self.solver.class_loss_weights)
        num_classes = len(self.data.class_names)
        if num_weights != num_classes:
            raise ConfigError(
                f'class_loss_weights ({num_weights}) must be same length as '
                f'the number of classes ({num_classes}), null class included')
