from typing import Any, Optional, Tuple

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import ObjectDetectionLabels
from rastervision.core.data.label_source import LabelSource
from rastervision.core.data.vector_source import VectorSource


class ObjectDetectionLabelSource(LabelSource):
    """A read-only label source for object detection."""

    def __init__(self,
                 vector_source: VectorSource,
                 extent: Box,
                 ioa_thresh: Optional[float] = None,
                 clip: bool = False):
        """Constructor.

        Args:
            vector_source (VectorSource): A VectorSource.
            extent (Box): Box used to filter the labels by extent.
            ioa_thresh (Optional[float], optional): IOA threshold to apply when
                retieving labels for a window. Defaults to None.
            clip (bool, optional): Clip bounding boxes to window limits when
                retrieving labels for a window. Defaults to False.
        """
        geojson = vector_source.get_geojson()
        self.validate_geojson(geojson)
        self.labels = ObjectDetectionLabels.from_geojson(
            geojson, extent=extent)
        self._extent = extent
        self.ioa_thresh = ioa_thresh if ioa_thresh is not None else 1e-6
        self.clip = clip

    def get_labels(self,
                   window: Box = None,
                   ioa_thresh: float = 1e-6,
                   clip: bool = False) -> ObjectDetectionLabels:
        """Get labels (in global coords) for a window.

        Args:
            window (Box): Window coords.

        Returns:
            ObjectDetectionLabels: Labels with sufficient overlap with the
                window. The returned labels are in global coods
                (i.e. coords wihtin the full extent).
        """
        if window is None:
            return self.labels
        window = window.shift_origin(self.extent)
        return ObjectDetectionLabels.get_overlapping(
            self.labels, window, ioa_thresh=ioa_thresh, clip=clip)

    def __getitem__(self, key: Any) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get labels (in window coords) for a window.

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
            Tuple[np.ndarray, np.ndarray, str]: 3-tuple of
                (npboxes, class_ids, box_format). The returned npboxes are in
                window coords (i.e. coords within the window).
        """
        if isinstance(key, Box):
            window = key
            labels = self.get_labels(
                window, ioa_thresh=self.ioa_thresh, clip=self.clip)
            class_ids = labels.get_class_ids()
            npboxes = labels.get_npboxes()
            npboxes = ObjectDetectionLabels.global_to_local(npboxes, window)
            return npboxes, class_ids, 'yxyx'
        elif isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            pass
        else:
            raise TypeError('Unsupported key type.')
        slices = list(key)
        assert 1 <= len(slices) <= 2
        assert all(s is not None for s in slices)
        assert isinstance(slices[0], slice)
        if len(slices) == 1:
            h, = slices
            w = slice(None, None)
        else:
            assert isinstance(slices[1], slice)
            h, w = slices

        if any(x is not None and x < 0
               for x in [h.start, h.stop, w.start, w.stop]):
            raise NotImplementedError()

        ymin, xmin, ymax, xmax = self.extent
        _ymin = 0 if h.start is None else h.start
        _xmin = 0 if w.start is None else w.start
        _ymax = ymax if h.stop is None else h.stop
        _xmax = xmax if w.stop is None else w.stop
        window = Box(_ymin, _xmin, _ymax, _xmax)

        npboxes, class_ids, fmt = self[window]

        # rescale if steps specified
        if h.step is not None:
            # assume fmt='yxyx'
            npboxes[:, [0, 2]] /= h.step
        if w.step is not None:
            # assume fmt='yxyx'
            npboxes[:, [1, 3]] /= w.step

        return npboxes, class_ids, fmt

    def validate_geojson(self, geojson: dict) -> None:
        for f in geojson['features']:
            geom_type = f.get('geometry', {}).get('type', '')
            if 'Point' in geom_type or 'LineString' in geom_type:
                raise ValueError(
                    'LineStrings and Points are not supported '
                    'in ChipClassificationLabelSource. Use BufferTransformer '
                    'to buffer them into Polygons.')
        for f in geojson['features']:
            if f.get('properties', {}).get('class_id') is None:
                raise ValueError('All GeoJSON features must have a class_id '
                                 'field in their properties.')

    @property
    def extent(self) -> Box:
        return self._extent
