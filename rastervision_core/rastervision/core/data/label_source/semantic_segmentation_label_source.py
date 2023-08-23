from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import ClassConfig
from rastervision.core.data.label import SemanticSegmentationLabels
from rastervision.core.data.label_source.label_source import LabelSource
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.utils import pad_to_window_size

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer


class SemanticSegmentationLabelSource(LabelSource):
    """A read-only label source for semantic segmentation."""

    def __init__(self,
                 raster_source: RasterSource,
                 class_config: ClassConfig,
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            raster_source (RasterSource): A raster source that returns a single
                channel raster with class_ids as values.
            null_class_id (int): the null class id used as fill values for when
                windows go over the edge of the label array. This can be
                retrieved using class_config.null_class_id.
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If None, the full extent available in the source file is used.
        """
        self.raster_source = raster_source
        self.class_config = class_config
        if bbox is not None:
            self.set_bbox(bbox)

    def enough_target_pixels(self, window: Box, target_count_threshold: int,
                             target_classes: List[int]) -> bool:
        """Check if window contains enough pixels of the given target classes.

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
        label_arr = self.get_label_arr(window)

        target_count = 0
        for class_id in target_classes:
            target_count += (label_arr == class_id).sum()

        return target_count >= target_count_threshold

    def get_labels(self,
                   window: Optional[Box] = None) -> SemanticSegmentationLabels:
        """Get labels for a window.

        Args:
            window (Optional[Box], optional): Window to get labels for. If
            None, returns labels covering the full extent of the scene.

        Returns:
            SemanticSegmentationLabels: The labels.
        """
        if window is None:
            window = self.extent

        label_arr = self.get_label_arr(window)
        labels = SemanticSegmentationLabels.make_empty(
            extent=self.extent,
            num_classes=len(self.class_config),
            smooth=False)
        labels[window] = label_arr

        return labels

    def get_label_arr(self, window: Optional[Box] = None) -> np.ndarray:
        """Get labels for a window.

        The returned array will be the same size as the input window. If window
        overflows the extent, the overflowing region will be filled with the ID
        of the null class as defined by the class_config.

        Args:
            window (Optional[Box], optional): Window (in pixel coords) to get
                labels for. If None, returns a label array covering the full
                extent of the scene.

        Returns:
            np.ndarray: Label array.
        """
        if window is None:
            window = self.extent

        label_arr = self.raster_source.get_chip(window)
        if label_arr.ndim == 3:
            label_arr = np.squeeze(label_arr, axis=2)
        h, w = label_arr.shape
        if h < window.height or w < window.width:
            label_arr = pad_to_window_size(label_arr, window, self.extent,
                                           self.class_config.null_class_id)
        return label_arr

    @property
    def bbox(self) -> Box:
        return self.raster_source.bbox

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.raster_source.crs_transformer

    def set_bbox(self, bbox: 'Box') -> None:
        self.raster_source.set_bbox(bbox)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, Box):
            return self.get_label_arr(key)
        else:
            return super().__getitem__(key)
