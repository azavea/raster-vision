from typing import TYPE_CHECKING, Callable, Iterable, Optional, Union
from enum import Enum
import logging

import albumentations as A
from torch import nn

from rastervision.core.data import Scene
from rastervision.core.rv_pipeline import WindowSamplingMethod
from rastervision.pipeline.config import (Config, register_config, ConfigError)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, ImageDataConfig, GeoDataConfig)
from rastervision.pytorch_learner.dataset import (
    ClassificationImageDataset, ClassificationSlidingWindowGeoDataset,
    ClassificationRandomWindowGeoDataset)
from rastervision.pytorch_learner.utils import adjust_conv_channels

if TYPE_CHECKING:
    from rastervision.core.data import SceneConfig

log = logging.getLogger(__name__)


class ClassificationDataFormat(Enum):
    image_folder = 'image_folder'


def clf_data_config_upgrader(cfg_dict, version):
    if version == 1:
        cfg_dict['type_hint'] = 'classification_image_data'
    return cfg_dict


@register_config('classification_data', upgrader=clf_data_config_upgrader)
class ClassificationDataConfig(Config):
    pass


@register_config('classification_image_data')
class ClassificationImageDataConfig(ClassificationDataConfig, ImageDataConfig):
    """Configure :class:`ClassificationImageDatasets <.ClassificationImageDataset>`."""

    data_format: ClassificationDataFormat = ClassificationDataFormat.image_folder

    def dir_to_dataset(self, data_dir: str, transform: A.BasicTransform
                       ) -> ClassificationImageDataset:
        ds = ClassificationImageDataset(
            data_dir, class_names=self.class_names, transform=transform)
        return ds


@register_config('classification_geo_data')
class ClassificationGeoDataConfig(ClassificationDataConfig, GeoDataConfig):
    """Configure classification :class:`GeoDatasets <.GeoDataset>`.

    See :mod:`rastervision.pytorch_learner.dataset.classification_dataset`.
    """

    def build_scenes(self,
                     scene_configs: Iterable['SceneConfig'],
                     tmp_dir: Optional[str] = None):
        for s in scene_configs:
            if s.label_source is not None:
                s.label_source.lazy = True
        return super().build_scenes(scene_configs, tmp_dir=tmp_dir)

    def scene_to_dataset(self,
                         scene: Scene,
                         transform: Optional[A.BasicTransform] = None,
                         for_chipping: bool = False
                         ) -> Union[ClassificationSlidingWindowGeoDataset,
                                    ClassificationRandomWindowGeoDataset]:
        if isinstance(self.sampling, dict):
            opts = self.sampling[scene.id]
        else:
            opts = self.sampling

        extra_args = {}
        if for_chipping:
            extra_args = dict(
                normalize=False, to_pytorch=False, return_window=True)

        if opts.method == WindowSamplingMethod.sliding:
            ds = ClassificationSlidingWindowGeoDataset(
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
            ds = ClassificationRandomWindowGeoDataset(
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


@register_config('classification_model')
class ClassificationModelConfig(ModelConfig):
    """Configure a classification model."""

    def build_default_model(self, num_classes: int,
                            in_channels: int) -> nn.Module:
        from torchvision import models

        backbone_name = self.get_backbone_str()
        pretrained = self.pretrained
        weights = 'DEFAULT' if pretrained else None
        model_factory_func: Callable[..., nn.Module] = getattr(
            models, backbone_name)
        model = model_factory_func(weights=weights, **self.extra_args)

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
            model.conv1 = adjust_conv_channels(
                old_conv=model.conv1,
                in_channels=in_channels,
                pretrained=pretrained)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        return model


@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    """Configure a :class:`.ClassificationLearner`."""

    model: Optional[ClassificationModelConfig]

    def build(self,
              tmp_dir=None,
              model_weights_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(
            self,
            tmp_dir=tmp_dir,
            model_weights_path=model_weights_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)
