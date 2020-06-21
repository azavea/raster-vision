from typing import (List, Optional)

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.class_config import ClassConfig
from rastervision.core.data import ActivateMixin
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_source.label_source import (LabelSource)
from rastervision.core.data.label_source.segmentation_class_transformer import (
    SegmentationClassTransformer)
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


class SemanticSegmentationLabelSource(ActivateMixin, LabelSource):
    """A read-only label source for semantic segmentation."""

    def __init__(self,
                 raster_source: RasterSource,
                 null_class_id: int,
                 rgb_class_config: ClassConfig = None):
        """Constructor.

        Args:
            raster_source: (RasterSource) A raster source that returns a single channel
                raster with class_ids as values, or a 3 channel raster with
                RGB values that are mapped to class_ids using the rgb_class_map
            null_class_id: (int) the null class id used as fill values for when windows
                go over the edge of the label array. This can be retrieved using
                class_config.get_null_class_id().
            rgb_class_config: (ClassConfig) with color values filled in.
                Optional and used to
                transform RGB values to class ids. Only use if the raster source
                is RGB.
        """
        self.raster_source = raster_source
        self.null_class_id = null_class_id
        self.class_transformer = None
        if rgb_class_config is not None:
            self.class_transformer = SegmentationClassTransformer(
                rgb_class_config)

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

        if self.class_transformer is not None:
            labels = self.class_transformer.rgb_to_class(raw_labels)
        else:
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
        labels = SemanticSegmentationLabels()
        window = window or self.raster_source.get_extent()
        raw_labels = self.raster_source.get_raw_chip(window)
        label_arr = (np.squeeze(raw_labels) if self.class_transformer is None
                     else self.class_transformer.rgb_to_class(raw_labels))

        label_arr = fill_edge(label_arr, window,
                              self.raster_source.get_extent(),
                              self.null_class_id)
        labels.set_label_arr(window, label_arr)
        return labels

    def _subcomponents_to_activate(self):
        return [self.raster_source]

    def _activate(self):
        pass

    def _deactivate(self):
        pass
