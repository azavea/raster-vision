import logging

from rastervision.pytorch_learner.object_detection_utils import CocoDataset
from rastervision.pytorch_learner.dataset import (TransformType, ImageDataset,
                                                  SlidingWindowGeoDataset,
                                                  RandomWindowGeoDataset)

log = logging.getLogger(__name__)


class ObjectDetectionImageDataset(ImageDataset):
    def __init__(self, img_dir, annotation_uri, transform=None):
        coco_ds = CocoDataset(img_dir, annotation_uri)
        super().__init__(
            orig_dataset=coco_ds,
            transform=transform,
            transform_type=TransformType.object_detection)


class ObjectDetectionSlidingWindowGeoDataset(SlidingWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)


class ObjectDetectionRandomWindowGeoDataset(RandomWindowGeoDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)
