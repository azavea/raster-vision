from typing import Optional, Tuple

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.core.data.label_source import LabelSource
from rastervision.core.data.vector_source import VectorSource


class ObjectDetectionLabelSource(LabelSource):
    """A read-only label source for object detection."""

    def __init__(self,
                 vector_source: VectorSource,
                 extent: Optional[Box] = None,
                 ioa_thresh: Optional[float] = None,
                 clip: bool = False):
        """Constructor.

        Args:
            vector_source (VectorSource): A VectorSource.
            extent (Optional[Box], optional): Box used to filter the labels by
                extent. Defaults to None.
            ioa_thresh (Optional[float], optional): IOA threshold to apply when
                retieving labels for a window. Defaults to None.
            clip (bool, optional): Clip bounding boxes to window limits when
                retrieving labels for a window. Defaults to False.
        """
        self.labels = ObjectDetectionLabels.from_geojson(
            vector_source.get_geojson(), extent=extent)
        self.ioa_thresh = ioa_thresh if ioa_thresh is not None else 1e-6
        self.clip = clip

    @property
    def ioa_thresh(self) -> float:
        return self._ioa_thresh

    @ioa_thresh.setter
    def ioa_thresh(self, value: float):
        self._ioa_thresh = value

    @property
    def clip(self) -> bool:
        return self._clip

    @clip.setter
    def clip(self, value: bool):
        self._clip = value

    def get_labels(self,
                   window: Box = None,
                   ioa_thresh: float = 1e-6,
                   clip: bool = False) -> ObjectDetectionLabels:
        if window is None:
            return self.labels
        return ObjectDetectionLabels.get_overlapping(
            self.labels, window, ioa_thresh=ioa_thresh, clip=clip)

    def __getitem__(self, window: Box) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get labels for a window.

        Returns a 3-tuple: (npboxes, class_ids, box_format).
        - npboxes is a float np.ndarray of shape (num_boxes, 4) representing
        pixel coords of bounding boxes in the form [ymin, xmin, ymax, xmax].
        - class_ids is a np.ndarray of shape (num_boxes,) representing the
        class labels for each of the boxes.
        - box_format is the format of npboxes which, in this case, is always
        'yxyx'.

        Args:
            window (Box): Window coords.

        Returns:
            Tuple[np.ndarray, np.ndarray, str]: [description]
        """
        labels = self.get_labels(
            window, ioa_thresh=self.ioa_thresh, clip=self.clip)
        class_ids = labels.get_class_ids()
        npboxes = labels.get_npboxes()
        npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
        return npboxes, class_ids, 'yxyx'
