from typing import (List, Optional)

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.raster_source import RasterSource


def fill_edge(label_arr, window, extent, fill_value):
    """If window goes over the edge of the extent, buffer with fill_value."""
    if window.ymax <= extent.ymax and window.xmax <= extent.xmax:
        return label_arr

    x = np.full((window.get_height(), window.get_width()), fill_value)
    ylim = extent.ymax - window.ymin
    xlim = extent.xmax - window.xmin
    x[0:ylim, 0:xlim] = label_arr[0:ylim, 0:xlim]
    return x


class SemanticSegmentationLabelSource(LabelSource):
    """A read-only label source for semantic segmentation."""

    def __init__(self, raster_source: RasterSource, null_class_id: int):
        """Constructor.

        Args:
            raster_source: (RasterSource) A raster source that returns a single channel
                raster with class_ids as values, or a 3 channel raster with
                RGB values that are mapped to class_ids using the rgb_class_map
            null_class_id: (int) the null class id used as fill values for when windows
                go over the edge of the label array. This can be retrieved using
                class_config.get_null_class_id().
        """
        self.raster_source = raster_source
        self.null_class_id = null_class_id

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
        raw_labels = self.raster_source.get_raw_chip(window)
        labels = np.squeeze(raw_labels)
        labels = fill_edge(labels, window, self.raster_source.get_extent(),
                           self.null_class_id)

        target_count = 0
        for class_id in target_classes:
            target_count = target_count + (labels == class_id).sum()

        return target_count >= target_count_threshold

    def get_labels(self,
                   window: Optional[Box] = None) -> SemanticSegmentationLabels:
        """Get labels for a window.

        Args:
             window: Either None or a window given as a Box object. Uses full extent of
                scene if window is not provided.
        Returns:
             SemanticSegmentationLabels
        """
        labels = SemanticSegmentationLabels.make_empty()
        window = window or self.raster_source.get_extent()
        raw_labels = self.raster_source.get_chip(window)
        label_arr = np.squeeze(raw_labels)
        label_arr = fill_edge(label_arr, window,
                              self.raster_source.get_extent(),
                              self.null_class_id)
        labels[window] = label_arr
        return labels

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_labels(window)[window]
