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

    def get_labels(self, window: Union[Box, None] = None,
                   chip_size=1000) -> SemanticSegmentationLabels:
        """Get labels for a window.

        To avoid running out of memory, if window is None and defaults to using the full
        extent, a set of sub-windows of size chip_size are used which cover the full
        extent with no overlap.

        Args:
             window: Either None or a window given as a Box object. Uses full extent of
                scene if window is not provided.
             chip_size: size of sub-windows to use if full extent is used.
        Returns:
             SemanticSegmentationLabels
        """

        def label_fn(_window):
            raw_labels = self.source.get_raw_chip(_window)

            if self.class_transformer is not None:
                labels = self.class_transformer.rgb_to_class(raw_labels)
            else:
                labels = np.squeeze(raw_labels)

            return labels

        windows = [window]
        if window is None:
            window = self.source.get_extent()
            windows = window.get_windows(chip_size, chip_size)

        return SemanticSegmentationLabels(windows, label_fn)

    def _subcomponents_to_activate(self):
        return [self.source]

    def _activate(self):
        pass

    def _deactivate(self):
        pass
