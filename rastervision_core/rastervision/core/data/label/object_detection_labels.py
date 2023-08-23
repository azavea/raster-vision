from typing import TYPE_CHECKING, Dict, Iterable, List, Optional
import numpy as np
from shapely.geometry import shape

from rastervision.core.box import Box
from rastervision.core.data.label.labels import Labels
from rastervision.core.data.label.tfod_utils.np_box_list import NpBoxList
from rastervision.core.data.label.tfod_utils.np_box_list_ops import (
    prune_non_overlapping_boxes, clip_to_window, concatenate,
    non_max_suppression)

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig, CRSTransformer)
    from shapely.geometry import Polygon


class ObjectDetectionLabels(Labels):
    """A set of boxes and associated class_ids and scores.

    Implemented using the Tensorflow Object Detection API's BoxList class.
    """

    def __init__(self,
                 npboxes: np.array,
                 class_ids: np.array,
                 scores: np.array = None):
        """Construct a set of object detection labels.

        Args:
            npboxes: float numpy array of size nx4 with cols
                ymin, xmin, ymax, xmax. Should be in pixel coordinates within
                the global frame of reference.
            class_ids: int numpy array of size n with class ids
            scores: float numpy array of size n
        """
        self.boxlist = NpBoxList(npboxes)
        # This field name actually needs to be 'classes' to be able to use
        # certain utility functions in the TF Object Detection API.
        self.boxlist.add_field('classes', class_ids)
        # We need to ensure that there is always a scores field so that the
        # concatenate method will work with empty labels objects.
        if scores is None:
            scores = np.ones(class_ids.shape)
        self.boxlist.add_field('scores', scores)

    def __add__(self,
                other: 'ObjectDetectionLabels') -> 'ObjectDetectionLabels':
        return ObjectDetectionLabels.concatenate(self, other)

    def __eq__(self, other: 'ObjectDetectionLabels') -> bool:
        return (isinstance(other, ObjectDetectionLabels)
                and self.to_dict() == other.to_dict())

    def __setitem__(self, window: Box, item: Dict[str, np.ndarray]):
        boxes = item['boxes']
        boxes = ObjectDetectionLabels.local_to_global(boxes, window)
        class_ids = item['class_ids']
        scores = item.get('scores')

        new_labels = ObjectDetectionLabels(boxes, class_ids, scores=scores)
        concatenated_labels = self + new_labels
        self.boxlist = concatenated_labels.boxlist

    def __getitem__(self, window: Box) -> 'ObjectDetectionLabels':
        return ObjectDetectionLabels.get_overlapping(self, window)

    def assert_equal(self, expected_labels: 'ObjectDetectionLabels'):
        np.testing.assert_array_equal(self.get_npboxes(),
                                      expected_labels.get_npboxes())
        np.testing.assert_array_equal(self.get_class_ids(),
                                      expected_labels.get_class_ids())
        np.testing.assert_array_equal(self.get_scores(),
                                      expected_labels.get_scores())

    def filter_by_aoi(self, aoi_polygons: Iterable['Polygon']):
        boxes = self.get_boxes()
        class_ids = self.get_class_ids()
        scores = self.get_scores()

        new_boxes = []
        new_class_ids = []
        new_scores = []
        for box, class_id, score in zip(boxes, class_ids, scores):
            box_poly = box.to_shapely()
            for aoi in aoi_polygons:
                if box_poly.within(aoi):
                    new_boxes.append(box.npbox_format())
                    new_class_ids.append(class_id)
                    new_scores.append(score)
                    break

        if len(new_boxes) == 0:
            return ObjectDetectionLabels.make_empty()

        return ObjectDetectionLabels(
            np.array(new_boxes), np.array(new_class_ids), np.array(new_scores))

    @classmethod
    def make_empty(cls) -> 'ObjectDetectionLabels':
        npboxes = np.empty((0, 4))
        class_ids = np.empty((0, ))
        scores = np.empty((0, ))
        return cls(npboxes, class_ids, scores)

    @staticmethod
    def from_boxlist(boxlist: NpBoxList):
        """Make ObjectDetectionLabels from BoxList object."""
        scores = (boxlist.get_field('scores')
                  if boxlist.has_field('scores') else None)
        return ObjectDetectionLabels(
            boxlist.get(), boxlist.get_field('classes'), scores=scores)

    @staticmethod
    def from_geojson(geojson: dict,
                     bbox: Optional[Box] = None,
                     ioa_thresh: float = 0.8,
                     clip: bool = True) -> 'ObjectDetectionLabels':
        """Convert GeoJSON to ObjectDetectionLabels object.

        If bbox is provided, filter out the boxes that lie "more than a little
        bit" outside the bbox.

        Args:
            geojson: (dict) normalized GeoJSON (see VectorSource)
            bbox: (Box) in pixel coords

        Returns:
            ObjectDetectionLabels
        """
        features = geojson['features']
        if len(features) == 0:
            labels = ObjectDetectionLabels.make_empty()
        else:
            boxes = [Box.from_shapely(shape(f['geometry'])) for f in features]
            class_ids = [f['properties']['class_id'] for f in features]
            scores = [f['properties'].get('score', 1.0) for f in features]

            boxes = np.array([b.npbox_format() for b in boxes], dtype=float)
            class_ids = np.array(class_ids)
            scores = np.array(scores)
            labels = ObjectDetectionLabels(boxes, class_ids, scores=scores)

        if bbox is not None:
            labels = ObjectDetectionLabels.get_overlapping(
                labels, bbox, ioa_thresh=ioa_thresh, clip=clip)
        return labels

    def get_boxes(self) -> List[Box]:
        """Return list of Boxes."""
        return [Box.from_npbox(npbox) for npbox in self.boxlist.get()]

    def get_npboxes(self) -> np.ndarray:
        return self.boxlist.get()

    def get_scores(self) -> np.ndarray:
        if self.boxlist.has_field('scores'):
            return self.boxlist.get_field('scores')
        return None

    def get_class_ids(self) -> np.ndarray:
        return self.boxlist.get_field('classes')

    def __len__(self) -> int:
        return self.boxlist.get().shape[0]

    def __str__(self) -> str:  # prama: no cover
        return str(self.boxlist.get())

    def to_boxlist(self) -> NpBoxList:
        return self.boxlist

    def to_dict(self, round_boxes: bool = True) -> dict:
        """Returns a dict version of these labels.

        The Dict has a Box as a key, and a tuple of (class_id, score)
        as the values.
        """
        npboxes = self.get_npboxes()
        if round_boxes and np.issubdtype(npboxes.dtype, np.floating):
            npboxes = npboxes.round(2)
        classes = self.get_class_ids()
        scores = self.get_scores().round(6)
        d = {
            Box.from_npbox(box): (class_id, score)
            for box, class_id, score in zip(npboxes, classes, scores)
        }
        return d

    @staticmethod
    def local_to_global(npboxes: np.ndarray, window: Box):
        """Convert from local to global coordinates.

        The local coordinates are row/col within the window frame of reference.
        The global coordinates are row/col within the extent of a RasterSource.
        """
        xmin = window.xmin
        ymin = window.ymin
        return npboxes + np.array([[ymin, xmin, ymin, xmin]])

    @staticmethod
    def global_to_local(npboxes: np.ndarray, window: Box):
        """Convert from global to local coordinates.

        The global coordinates are row/col within the extent of a RasterSource.
        The local coordinates are row/col within the window frame of reference.
        """
        xmin = window.xmin
        ymin = window.ymin
        return npboxes - np.array([[ymin, xmin, ymin, xmin]])

    @staticmethod
    def local_to_normalized(npboxes: np.ndarray, window: Box):
        """Convert from local to normalized coordinates.

        The local coordinates are row/col within the window frame of reference.
        Normalized coordinates range from 0 to 1 on each (height/width) axis.
        """
        height, width = window.size
        return npboxes / np.array([[height, width, height, width]])

    @staticmethod
    def normalized_to_local(npboxes: np.ndarray, window: Box):
        """Convert from normalized to local coordinates.

        Normalized coordinates range from 0 to 1 on each (height/width) axis.
        The local coordinates are row/col within the window frame of reference.
        """
        height, width = window.size
        return npboxes * np.array([[height, width, height, width]])

    @staticmethod
    def get_overlapping(labels: 'ObjectDetectionLabels',
                        window: Box,
                        ioa_thresh: float = 0.5,
                        clip: bool = False) -> 'ObjectDetectionLabels':
        """Return subset of labels that overlap with window.

        Args:
            labels: ObjectDetectionLabels
            window: Box
            ioa_thresh: The minimum intersection-over-area (IOA) for a box to
                be considered as overlapping. For each box, IOA is defined as
                the area of the intersection of the box with the window over
                the area of the box.
            clip: If True, clip label boxes to the window.
        """
        window_npbox = window.npbox_format()
        window_boxlist = NpBoxList(np.expand_dims(window_npbox, axis=0))
        boxlist = prune_non_overlapping_boxes(
            labels.boxlist, window_boxlist, minoverlap=ioa_thresh)
        if clip:
            boxlist = clip_to_window(boxlist, window_npbox)

        return ObjectDetectionLabels.from_boxlist(boxlist)

    @staticmethod
    def concatenate(
            labels1: 'ObjectDetectionLabels',
            labels2: 'ObjectDetectionLabels') -> 'ObjectDetectionLabels':
        """Return concatenation of labels.

        Args:
            labels1: ObjectDetectionLabels
            labels2: ObjectDetectionLabels
        """
        new_boxlist = concatenate([labels1.to_boxlist(), labels2.to_boxlist()])
        return ObjectDetectionLabels.from_boxlist(new_boxlist)

    @staticmethod
    def prune_duplicates(
            labels: 'ObjectDetectionLabels',
            score_thresh: float,
            merge_thresh: float,
            max_output_size: Optional[int] = None) -> 'ObjectDetectionLabels':
        """Remove duplicate boxes via non-maximum suppression.

        Args:
            labels: Labels whose boxes are to be pruned.
            score_thresh: Prune boxes with score less than this threshold.
            merge_thresh: Prune boxes with intersection-over-union (IOU)
                greater than this threshold.
            max_output_size (int): Maximum number of retained boxes.
                If None, this is set to ``len(abels)``. Defaults to None.

        Returns:
            ObjectDetectionLabels: Pruned labels.
        """
        if max_output_size is None:
            max_output_size = len(labels)
        pruned_boxlist = non_max_suppression(
            labels.boxlist,
            max_output_size=max_output_size,
            iou_threshold=merge_thresh,
            score_threshold=score_thresh)
        return ObjectDetectionLabels.from_boxlist(pruned_boxlist)

    def save(self, uri: str, class_config: 'ClassConfig',
             crs_transformer: 'CRSTransformer') -> None:
        """Save labels as a GeoJSON file.

        Args:
            uri (str): URI of the output file.
            class_config (ClassConfig): ClassConfig to map class IDs to names.
            crs_transformer (CRSTransformer): CRSTransformer to convert from
                pixel-coords to map-coords before saving.
        """
        from rastervision.core.data import ObjectDetectionGeoJSONStore

        label_store = ObjectDetectionGeoJSONStore(
            uri=uri,
            class_config=class_config,
            crs_transformer=crs_transformer)
        label_store.save(self)
