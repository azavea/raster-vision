from typing import (List, Union)

import numpy as np

from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.data import ActivateMixin
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.data.label_source import LabelSource, SegmentationClassTransformer
from rastervision.data.raster_source import RasterSource


class SemanticSegmentationLabelSource(ActivateMixin, LabelSource):
    """A read-only label source for semantic segmentation."""

    def __init__(self, source: RasterSource, rgb_class_map: ClassMap = None):
        """Constructor.

        Args:
            source: (RasterSource) A raster source that returns a single channel
                raster with class_ids as values, or a 3 channel raster with
                RGB values that are mapped to class_ids using the rgb_class_map
            rgb_class_map: (ClassMap) with color values filled in. Optional and used to
                transform RGB values to class ids. Only use if the raster source
                is RGB.
        """
        self.source = source
        self.class_transformer = None
        if rgb_class_map is not None:
            self.class_transformer = SegmentationClassTransformer(
                rgb_class_map)

    def enough_target_pixels(self, window: Box, target_count_threshold: int,
                             target_classes: List[int]) -> bool:
        """Given a window, answer whether the window contains enough pixels in
        the target classes.

        Args:
             window: The larger window from-which the sub-window will
                  be clipped.
             target_count_threshold:  Minimum number of target pixels.
             target_classes: The classes of interest.  The given
                  window is examined to make sure that it contains a
                  sufficient number of target pixels.
        Returns:
             True (the window does contain interesting pixels) or False.
        """
        raw_labels = self.source.get_raw_chip(window)
        if self.class_transformer is not None:
            labels = self.class_transformer.rgb_to_class(raw_labels)
        else:
            labels = np.squeeze(raw_labels)

        target_count = 0
        for class_id in target_classes:
            target_count = target_count + (labels == class_id).sum()

        return target_count >= target_count_threshold

    def get_labels(self, window: Union[Box, None] = None) -> np.ndarray:
        """Get labels from a window.

        If window is not None then a label window is clipped from
        the source. If window is None then assume window is full extent.

        Args:
             window: Either None or a window given as a Box object.
        Returns:
             SemanticSegmentationLabels covering window
        """
        if window is None:
            raw_labels = self.source.get_raw_image_array()
        else:
            raw_labels = self.source.get_raw_chip(window)

        if self.class_transformer is not None:
            labels = self.class_transformer.rgb_to_class(raw_labels)
        else:
            labels = np.squeeze(raw_labels)

        return SemanticSegmentationLabels.from_array(labels)

    def _subcomponents_to_activate(self):
        return [self.source]

    def _activate(self):
        pass

    def _deactivate(self):
        pass
