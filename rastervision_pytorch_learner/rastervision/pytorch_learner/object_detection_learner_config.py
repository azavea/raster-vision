from typing import TYPE_CHECKING, Optional, Union
from enum import Enum
from os.path import join
import logging

import albumentations as A

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from rastervision.core.data import Scene
from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator, ConfigError)
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, ModelConfig, Backbone, ImageDataConfig, GeoDataConfig,
    GeoDataWindowMethod, GeoDataWindowConfig)
from rastervision.pytorch_learner.dataset import (
    ObjectDetectionImageDataset, ObjectDetectionSlidingWindowGeoDataset,
    ObjectDetectionRandomWindowGeoDataset)
from rastervision.pytorch_learner.utils import adjust_conv_channels
from torchvision.models.detection.faster_rcnn import FasterRCNN

if TYPE_CHECKING:
    from rastervision.pytorch_learner.learner_config import SolverConfig

log = logging.getLogger(__name__)

DEFAULT_BBOX_PARAMS = A.BboxParams(
    format='albumentations', label_fields=['category_id'])


class ObjectDetectionDataFormat(Enum):
    coco = 'coco'


def objdet_data_config_upgrader(cfg_dict, version):
    if version == 1:
        cfg_dict['type_hint'] = 'object_detection_image_data'
    return cfg_dict


@register_config('object_detection_data', upgrader=objdet_data_config_upgrader)
class ObjectDetectionDataConfig(Config):
    def get_bbox_params(self):
        return DEFAULT_BBOX_PARAMS


@register_config('object_detection_image_data')
class ObjectDetectionImageDataConfig(ObjectDetectionDataConfig,
                                     ImageDataConfig):
    """Configure :class:`ObjectDetectionImageDatasets <.ObjectDetectionImageDataset>`."""

    data_format: ObjectDetectionDataFormat = ObjectDetectionDataFormat.coco

    def dir_to_dataset(self, data_dir: str, transform: A.BasicTransform
                       ) -> ObjectDetectionImageDataset:
        img_dir = join(data_dir, 'img')
        annotation_uri = join(data_dir, 'labels.json')
        ds = ObjectDetectionImageDataset(
            img_dir, annotation_uri, transform=transform)
        return ds


@register_config('object_detection_geo_data_window')
class ObjectDetectionGeoDataWindowConfig(GeoDataWindowConfig):
    """Configure an object detection :class:`.GeoDataset`.

    See :mod:`rastervision.pytorch_learner.dataset.object_detection_dataset`.
    """
    ioa_thresh: float = Field(
        0.8,
        description='When a box is partially outside of a training chip, it '
        'is not clear if (a clipped version) of the box should be included in '
        'the chip. If the IOA (intersection over area) of the box with the '
        'chip is greater than ioa_thresh, it is included in the chip. '
        'Defaults to 0.8.')
    clip: bool = Field(
        False,
        description='Clip bounding boxes to window limits when retrieving '
        'labels for a window.')
    neg_ratio: Optional[float] = Field(
        None,
        description='The ratio of negative chips (those containing no '
        'bounding boxes) to positive chips. This can be useful if the '
        'statistics of the background is different in positive chips. For '
        'example, in car detection, the positive chips will always contain '
        'roads, but no examples of rooftops since cars tend to not be near '
        'rooftops. Defaults to None.')
    neg_ioa_thresh: float = Field(
        0.2,
        description='A window will be considered negative if its max IoA with '
        'any bounding box is less than this threshold. Defaults to 0.2.')


@register_config('object_detection_geo_data')
class ObjectDetectionGeoDataConfig(ObjectDetectionDataConfig, GeoDataConfig):
    """Configure object detection :class:`GeoDatasets <.GeoDataset>`.

    See :mod:`rastervision.pytorch_learner.dataset.object_detection_dataset`.
    """

    def scene_to_dataset(
            self,
            scene: Scene,
            transform: Optional[A.BasicTransform] = None,
            bbox_params: Optional[A.BboxParams] = DEFAULT_BBOX_PARAMS
    ) -> Union[ObjectDetectionSlidingWindowGeoDataset,
               ObjectDetectionRandomWindowGeoDataset]:
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
                pad_direction=opts.pad_direction,
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
                max_sample_attempts=opts.max_sample_attempts,
                bbox_params=bbox_params,
                ioa_thresh=opts.ioa_thresh,
                clip=opts.clip,
                neg_ratio=opts.neg_ratio,
                neg_ioa_thresh=opts.neg_ioa_thresh,
                efficient_aoi_sampling=opts.efficient_aoi_sampling,
                transform=transform)
        else:
            raise NotImplementedError()
        return ds


@register_config('object_detection_model')
class ObjectDetectionModelConfig(ModelConfig):
    """Configure an object detection model."""

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

    def build_default_model(self, num_classes: int, in_channels: int,
                            img_sz: int) -> FasterRCNN:
        """Returns a FasterRCNN model.

        Note that the model returned will have (num_classes + 2) output
        classes. +1 for the null class (zeroth index), and another +1
        (last index) for backward compatibility with earlier Raster Vision
        versions.

        Returns:
            FasterRCNN: a FasterRCNN model.
        """
        pretrained = self.pretrained
        backbone_arch = self.get_backbone_str()
        backbone = resnet_fpn_backbone(backbone_arch, pretrained)

        # default values from FasterRCNN constructor
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]

        if in_channels != 3:
            extra_channels = in_channels - backbone.body['conv1'].in_channels

            # adjust channels
            backbone.body['conv1'] = adjust_conv_channels(
                old_conv=backbone.body['conv1'],
                in_channels=in_channels,
                pretrained=pretrained)

            # adjust stats
            if extra_channels < 0:
                image_mean = image_mean[:extra_channels]
                image_std = image_std[:extra_channels]
            else:
                # arbitrarily set mean and stds of the new channels to
                # something similar to the values of the other 3 channels
                image_mean = image_mean + [.45] * extra_channels
                image_std = image_std + [.225] * extra_channels

        model = FasterRCNN(
            backbone=backbone,
            # +1 because torchvision detection models reserve 0 for the null
            # class, another +1 for backward compatibility with earlier Raster
            # Vision versions
            num_classes=num_classes + 1 + 1,
            # TODO we shouldn't need to pass the image size here
            min_size=img_sz,
            max_size=img_sz,
            image_mean=image_mean,
            image_std=image_std)
        return model


@register_config('object_detection_learner')
class ObjectDetectionLearnerConfig(LearnerConfig):
    """Configure an :class:`.ObjectDetectionLearner`."""

    data: Union[ObjectDetectionImageDataConfig, ObjectDetectionGeoDataConfig]
    model: Optional[ObjectDetectionModelConfig]

    def build(self,
              tmp_dir=None,
              model_weights_path=None,
              model_def_path=None,
              loss_def_path=None,
              training=True):
        from rastervision.pytorch_learner.object_detection_learner import (
            ObjectDetectionLearner)
        return ObjectDetectionLearner(
            self,
            tmp_dir=tmp_dir,
            model_weights_path=model_weights_path,
            model_def_path=model_def_path,
            loss_def_path=loss_def_path,
            training=training)

    @validator('solver')
    def validate_solver_config(cls, v: 'SolverConfig') -> 'SolverConfig':
        if v.ignore_class_index is not None:
            raise ConfigError(
                'ignore_last_class is not supported for Object Detection.')
        if v.class_loss_weights is not None:
            raise ConfigError(
                'class_loss_weights is currently not supported for '
                'Object Detection.')
        if v.external_loss_def is not None:
            raise ConfigError(
                'external_loss_def is currently not supported for '
                'Object Detection. Raster Vision expects object '
                'detection models to behave like TorchVision object detection '
                'models, and these models compute losses internally. So, if '
                'you want to use a custom loss function, you can create a '
                'custom model that implements that loss function and use that '
                'model via external_model_def. See cowc_potsdam.py for an '
                'example of how to use a custom object detection model.')
        return v
