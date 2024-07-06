from typing import TYPE_CHECKING, Any

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
                 bbox: Box | None = None):
        """Constructor.

        Args:
            raster_source: A raster source that returns a single channel raster
                with class_ids as values.
            null_class_id: the null class id used as fill values for when
                windows go over the edge of the label array. This can be
                retrieved using ``class_config.null_class_id``.
            bbox: User-specified crop of the extent. If ``None``, the full
                extent available in the source file is used.
        """
        self.raster_source = raster_source
        self.class_config = class_config
        if bbox is not None:
            self.set_bbox(bbox)

    def get_labels(self,
                   window: Box | None = None) -> SemanticSegmentationLabels:
        """Get labels for a window.

        Args:
            window: Window to get labels for. If None, returns labels covering
                the full extent of the scene.

        Returns:
            The labels.
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

    def get_label_arr(self, window: Box | None = None) -> np.ndarray:
        """Get labels for a window.

        The returned array will be the same size as the input window. If window
        overflows the extent, the overflowing region will be filled with the ID
        of the null class as defined by the class_config.

        Args:
            window: Window (in pixel coords) to get labels for. If ``None``,
                returns a label array covering the full extent of the scene.

        Returns:
            Label array.
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
