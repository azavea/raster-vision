from enum import Enum
from typing import Union, Optional
from os.path import join

import albumentations as A

from torch.utils.data import Dataset

from rastervision.core.data import Scene
from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, Backbone, ImageDataConfig, GeoDataConfig,
    GeoDataWindowMethod)
from rastervision.pytorch_learner.dataset import (
    ObjectDetectionImageDataset, ObjectDetectionSlidingWindowGeoDataset,
    ObjectDetectionRandomWindowGeoDataset)


class ObjectDetectionDataFormat(Enum):
    coco = 'coco'


def objdet_data_config_upgrader(cfg_dict, version):
    if version == 1:
        cfg_dict['type_hint'] = 'object_detection_image_data'
    return cfg_dict


@register_config('object_detection_data', upgrader=objdet_data_config_upgrader)
class ObjectDetectionDataConfig(Config):
    pass


@register_config('object_detection_image_data')
class ObjectDetectionImageDataConfig(ObjectDetectionDataConfig,
                                     ImageDataConfig):
    data_format: ObjectDetectionDataFormat = ObjectDetectionDataFormat.coco

    def dir_to_dataset(self, data_dir: str,
                       transform: A.BasicTransform) -> Dataset:
        img_dir = join(data_dir, 'img')
        annotation_uri = join(data_dir, 'labels.json')
        ds = ObjectDetectionImageDataset(
            img_dir, annotation_uri, transform=transform)
        return ds


@register_config('object_detection_geo_data')
class ObjectDetectionGeoDataConfig(ObjectDetectionDataConfig, GeoDataConfig):
    def scene_to_dataset(
            self,
            scene: Scene,
            transform: Optional[A.BasicTransform] = None,
            bbox_params: Optional[A.BboxParams] = None) -> Dataset:
        if isinstance(self.window_opts, dict):
            opts = self.window_opts[scene.id]
        else:
            opts = self.window_opts

        if opts.method == GeoDataWindowMethod.sliding:
            ds = ObjectDetectionSlidingWindowGeoDataset(
                scene,
                size=opts.size,
                stride=opts.stride,
                padding=opts.padding,
                transform=transform)
        elif opts.method == GeoDataWindowMethod.random:
            ds = ObjectDetectionRandomWindowGeoDataset(
                scene,
                size_lims=opts.size_lims,
                h_lims=opts.h_lims,
                w_lims=opts.w_lims,
                out_size=opts.size,
                padding=opts.padding,
                max_windows=opts.max_windows,
                transform=transform,
                bbox_params=bbox_params)
        else:
            raise NotImplementedError()
        return ds


@register_config('object_detection_model')
class ObjectDetectionModelConfig(ModelConfig):
    backbone: Backbone = Field(
        Backbone.resnet50,
        description=
        ('The torchvision.models backbone to use, which must be in the resnet* '
         'family.'))

    @validator('backbone')
    def only_valid_backbones(cls, v):
        if v not in [
                Backbone.resnet18, Backbone.resnet34, Backbone.resnet50,
                Backbone.resnet101, Backbone.resnet152
        ]:
            raise ValueError(
                'The backbone for Faster-RCNN must be in the resnet* '
                'family.')
        return v


@register_config('object_detection_learner')
class ObjectDetectionLearnerConfig(LearnerConfig):
    data: Union[ObjectDetectionImageDataConfig, ObjectDetectionGeoDataConfig]
    model: ObjectDetectionModelConfig

    def build(self,
              tmp_dir,
              model_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.object_detection_learner import (
            ObjectDetectionLearner)
        return ObjectDetectionLearner(
            self,
            tmp_dir=tmp_dir,
            model_path=model_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)
