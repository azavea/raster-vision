from typing import Optional
import logging

import albumentations as A

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
        """Constructor.

        Args:
            bbox_params (Optional[A.BboxParams], optional): Optional
                bbox_params to use when resizing windows. Defaults to None.
        """
        self.bbox_params: Optional[A.BboxParams] = kwargs.pop(
            'bbox_params', None)

        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)

    def get_resize_transform(self, transform, out_size):
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose(
                [transform, resize_tf], bbox_params=self.bbox_params)
        return transform
