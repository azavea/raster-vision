from typing import Optional
import logging

import albumentations as A
import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import ObjectDetectionLabels
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

        Keyword Args:
            bbox_params (Optional[A.BboxParams], optional): Optional
                bbox_params to use when resizing windows. Defaults to None.
            ioa_thresh (float, optional): Minimum IoA of a bounding box with a
                given window for it to be included in the labels for that
                window. Defaults to 0.9.
            clip (bool, optional): Clip bounding boxes to window limits when
                retrieving labels for a window. Defaults to False.
            neg_ratio (Optional[float], optional): Ratio of sampling
                probabilities of negative windows (windows w/o bboxes) vs
                positive windows (windows w/ at least 1 bbox). E.g. neg_ratio=2
                means 2/3 probability of sampling a negative window.
                If None, the default sampling behavior of
                RandomWindowGeoDataset is used, without taking bboxes into
                account. Defaults to None.
            neg_ioa_thresh (float, optional): A window will be considered
                negative if its max IoA with any bounding box is less than this
                threshold. Defaults to 0.2.
        """
        self.bbox_params: Optional[A.BboxParams] = kwargs.pop(
            'bbox_params', None)
        ioa_thresh: float = kwargs.pop('ioa_thresh', 0.9)
        clip: bool = kwargs.pop('clip', False)
        neg_ratio: Optional[float] = kwargs.pop('neg_ratio', None)
        neg_ioa_thresh: float = kwargs.pop('neg_ioa_thresh', 0.2)

        super().__init__(
            *args, **kwargs, transform_type=TransformType.object_detection)

        self.scene.label_source.ioa_thresh = ioa_thresh
        self.scene.label_source.clip = clip

        if neg_ratio is not None:
            self.neg_probability = neg_ratio / (neg_ratio + 1)
            self.neg_ioa_thresh: float = neg_ioa_thresh

            # Get labels for the entire scene.
            # clip=True here to ensure that any window we draw around a box
            # will always lie inside the scene.
            self.labels = self.scene.label_source.get_labels(
                ioa_thresh=ioa_thresh, clip=True)
            self.bboxes = self.labels.get_boxes()
            if len(self.bboxes) == 0:
                raise ValueError(
                    'neg_ratio specified, but no bboxes found in scene.')
        else:
            self.neg_probability = None

    def get_resize_transform(self, transform, out_size):
        resize_tf = A.Resize(*out_size, always_apply=True)
        if transform is None:
            transform = resize_tf
        else:
            transform = A.Compose(
                [transform, resize_tf], bbox_params=self.bbox_params)
        return transform

    def _sample_pos_window(self) -> Box:
        """Sample a window that contains at least one bounding box.
        This is done by randomly sampling one of the bounding boxes in the
        scene and drawing a random window around it.
        """
        bbox = np.random.choice(self.bboxes)
        box_h, box_w = bbox.size

        # check if it is possible to sample a containing widnow
        hmax, wmax = self.max_size
        if box_h > hmax or box_w > wmax:
            raise ValueError(
                f'Cannot sample containing window because bounding box {bbox}'
                f'is larger than self.max_size ({self.max_size}).')

        # try to sample a window size that is larger than the box's size
        for _ in range(self.max_sample_attempts):
            h, w = self.sample_window_size()
            if h >= box_h and w >= box_w:
                window = bbox.make_random_box_container(h, w)
                return window
        log.warn('ObjectDetectionRandomWindowGeoDataset: Failed to find '
                 'suitable (h, w) for positive window. '
                 f'Using (hmax, wmax) = ({hmax}, {wmax}) instead.')
        window = bbox.make_random_box_container(hmax, wmax)
        return window

    def _sample_neg_window(self) -> Box:
        """Attempt to sample, within self.max_sample_attempts, a window
        containing no bounding boxes.
        If not found within self.max_sample_attempts, just return the last
        sampled window.
        """
        for _ in range(self.max_sample_attempts):
            window = super()._sample_window()
            labels = ObjectDetectionLabels.get_overlapping(
                self.labels, window, ioa_thresh=self.neg_ioa_thresh)
            if len(labels) == 0:
                return window

        log.warn('ObjectDetectionRandomWindowGeoDataset: Failed to find '
                 'negative window. Returning last sampled window.')
        return window

    def _sample_window(self) -> Box:
        """If self.neg_probability is specified, sample a negative or positive window
        based on that probability. Otherwise, just use RandomWindowGeoDataset's
        default window sampling behavior.
        """
        if self.neg_probability is None:
            return super()._sample_window()

        if np.random.sample() < self.neg_probability:
            return self._sample_neg_window()
        return self._sample_pos_window()
