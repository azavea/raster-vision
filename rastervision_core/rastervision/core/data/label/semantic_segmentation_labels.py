from abc import abstractmethod
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision.core.box import Box
from rastervision.core.data.label import Labels


class SemanticSegmentationLabels(Labels):
    """Representation of Semantic Segmentation labels."""

    def __init__(self, extent: Box, num_classes: int, dtype: np.dtype):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
        """
        self.extent = extent
        self.num_classes = num_classes
        self.ymin, self.xmin, self.width, self.height = extent.to_xywh()
        self.dtype = dtype

    def _to_local_coords(self, window: Union[Box, Tuple[int, int, int, int]]
                         ) -> Tuple[int, int, int, int]:
        """Convert window to extent coordinates.

        Args:
            window (Union[Box, Tuple[int, int, int, int]]): A rastervision
                Box or a 4-tuple of (ymin, xmin, ymax, xmax).

        Returns:
            Tuple[int, int, int, int]: a 4-tuple of (ymin, xmin, ymax, xmax).
        """
        ymin, xmin, ymax, xmax = window
        # convert to extent coords
        ymin_local, xmin_local = ymin - self.ymin, xmin - self.xmin
        ymax_local, xmax_local = ymax - self.ymin, xmax - self.xmin
        # clip negative values to zero
        ymin_local, xmin_local = max(0, ymin_local), max(0, xmin_local)
        ymax_local, xmax_local = max(0, ymax_local), max(0, xmax_local)
        # clip max values to array bounds
        ymax_local = min(self.height, ymax_local)
        xmax_local = min(self.width, xmax_local)
        return ymin_local, xmin_local, ymax_local, xmax_local

    @abstractmethod
    def __add__(self, other) -> 'SemanticSegmentationLabels':
        """Merge self with other labels."""
        pass

    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        """Set labels for the given window."""
        self.add_window(window, values)

    @abstractmethod
    def __delitem__(self, window: Box) -> None:
        """Delete labels for the given window."""
        pass

    @abstractmethod
    def __getitem__(self, window: Box) -> np.ndarray:
        """Get labels for the given window."""
        pass

    @abstractmethod
    def add_window(self, window: Box, values: np.ndarray) -> List[Box]:
        """Set labels for the given window."""
        pass

    @abstractmethod
    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Note: The returned array is not guaranteed to be the same size as the
        input window.
        """
        pass

    def get_windows(self, **kwargs) -> List[Box]:
        """Generate sliding windows over the local extent. The keyword args
        are passed to Box.get_windows() and can therefore be used to control
        the specifications of the windows.

        If the keyword args do not contain size, a list of length 1,
        containing the full extent is returned.
        """
        size: Optional[int] = kwargs.pop('size', None)
        if size is None:
            return [self.extent]
        return self.extent.get_windows(size, size, **kwargs)

    def filter_by_aoi(self, aoi_polygons: list, null_class_id: int,
                      **kwargs) -> 'SemanticSegmentationLabels':
        """Keep only the values that lie inside the AOI."""
        if not aoi_polygons:
            return self
        for window in self.get_windows(**kwargs):
            self._filter_window_by_aoi(window, aoi_polygons, null_class_id)
        return self

    @abstractmethod
    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Given a window and a binary mask, set all the pixels in the window
        for which the mask is ON to the fill_value.
        """
        pass

    def _filter_window_by_aoi(self, window: Box, aoi_polygons: list,
                              null_class_id: int) -> None:
        window_geom = window.to_shapely()
        label_arr = self[window]

        # For each aoi_polygon, intersect with window, and
        # put in window frame of reference.
        window_aois = []
        for aoi in aoi_polygons:
            window_aoi = aoi.intersection(window_geom)
            if not window_aoi.is_empty:

                def transform_shape(x, y, z=None):
                    return (x - window.xmin, y - window.ymin)

                window_aoi = transform(transform_shape, window_aoi)
                window_aois.append(window_aoi)

        # If window does't overlap with any AOI, then it won't be in
        # new_labels.
        if window_aois:
            # If window intersects with AOI, set pixels outside the
            # AOI polygon to 0 so they are ignored during eval.
            mask = rasterize(
                [(p, 0) for p in window_aois],
                out_shape=label_arr.shape[-2:],
                fill=1,
                dtype=np.uint8)
            mask = mask.astype(bool)
            self.mask_fill(window, mask, null_class_id)
        else:
            del self[window]

    @classmethod
    def make_empty(cls, extent: Box, num_classes: int, smooth: bool = False
                   ) -> Union['SemanticSegmentationDiscreteLabels',
                              'SemanticSegmentationSmoothLabels']:
        """Instantiate an empty instance.

        Args:
            extent (Box): The extent of the region to which the labels belong,
                in global coordinates.
            num_classes (int): Number of classes.
            smooth (bool, optional): If True, creates a
                SemanticSegmentationSmoothLabels object. If False, creates a
                SemanticSegmentationDiscreteLabels object. Defaults to False.

        Returns:
            Union[SemanticSegmentationDiscreteLabels,
            SemanticSegmentationSmoothLabels]: If smooth=True, returns a
                SemanticSegmentationSmoothLabels. Otherwise, a
                SemanticSegmentationDiscreteLabels.

        Raises:
            ValueError: if num_classes and extent are not specified, but
                smooth=True.
        """
        if not smooth:
            return SemanticSegmentationDiscreteLabels.make_empty(
                extent=extent, num_classes=num_classes)
        else:
            return SemanticSegmentationSmoothLabels.make_empty(
                extent=extent, num_classes=num_classes)

    @classmethod
    def from_predictions(cls,
                         windows: Iterable['Box'],
                         predictions: Iterable[Any],
                         extent: Optional[Box] = None,
                         num_classes: Optional[int] = None,
                         smooth: bool = False
                         ) -> Union['SemanticSegmentationDiscreteLabels',
                                    'SemanticSegmentationSmoothLabels']:
        """Instantiate from windows and their corresponding predictions.

        Args:
            windows (Iterable[Box]): Boxes in pixel coords, specifying chips
                in the raster.
            predictions (Iterable[Any]): The model predictions for each chip
                specified by the windows.
            extent (Box): The extent of the region to which the labels belong,
                in global coordinates.
            num_classes (int): Number of classes.

        Returns:
            SemanticSegmentationSmoothLabels: A
                SemanticSegmentationSmoothLabels instance populated with the
                given predictions.
        """
        labels = cls.make_empty(extent, num_classes, smooth=smooth)
        # If predictions is tqdm-wrapped, it needs to be the first arg to zip()
        # or the progress bar won't terminate with the correct count.
        for prediction, window in zip(predictions, windows):
            labels[window] = prediction
        return labels


class SemanticSegmentationDiscreteLabels(SemanticSegmentationLabels):
    """Vote-counts for each pixel belonging to each class.

    Maintains a num_classes x H x W array where value_{ijk} represents how
    many times pixel_{jk} has been classified as class i. A label array can be
    obtained from this by argmax'ing along the class dimension. Can also be
    turned into a score converting counts to probabilities.
    """

    def __init__(self, extent: Box, num_classes: int, dtype: Any = np.uint8):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
            dtype (Any): dtype of the counts array. Defaults to np.uint8.
        """
        super().__init__(extent, num_classes, dtype)

        self.pixel_counts = np.zeros(
            (self.num_classes, self.height, self.width), dtype=self.dtype)
        # track which pixels have been hit at all
        self.hit_mask = np.zeros((self.height, self.width), dtype=bool)

    def __add__(self, other: 'SemanticSegmentationDiscreteLabels'
                ) -> 'SemanticSegmentationDiscreteLabels':
        """Merge self with other labels by adding the pixel counts."""
        if self.extent != other.extent:
            raise ValueError('Cannot add labels with unqeual extents.')

        self.pixel_counts += other.pixel_counts
        return self

    def __eq__(self, other: 'SemanticSegmentationDiscreteLabels') -> bool:
        if not isinstance(other, SemanticSegmentationDiscreteLabels):
            return False
        if self.extent != other.extent:
            return False
        mask_equal = np.all(self.hit_mask == other.hit_mask)
        if not mask_equal:
            return False
        counts_equal = np.all(self.pixel_counts == other.pixel_counts)
        return counts_equal

    def __delitem__(self, window: Box) -> None:
        """Reset counts to zero for pixels in the window."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        self.pixel_counts[..., y0:y1, x0:x1] = 0
        self.hit_mask[y0:y1, x0:x1] = False

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_label_arr(window)

    def add_window(self, window: Box, pixel_class_ids: np.ndarray) -> None:
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        pixel_class_ids = pixel_class_ids.astype(self.dtype)
        pixel_class_ids = pixel_class_ids[..., :h, :w]
        window_pixel_counts = self.pixel_counts[:, y0:y1, x0:x1]
        for ch_class_id, ch in enumerate(window_pixel_counts):
            ch[pixel_class_ids == ch_class_id] += 1
        self.hit_mask[y0:y1, x0:x1] = True

    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Returns null_class_id for pixels for which there is no data.
        """
        y0, x0, y1, x1 = self._to_local_coords(window)
        label_arr = self.pixel_counts[..., y0:y1, x0:x1].argmax(axis=0)
        hit_mask = self.hit_mask[y0:y1, x0:x1]
        return np.where(hit_mask, label_arr, null_class_id)

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get array of pixel scores."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        class_counts = self.pixel_counts[..., y0:y1, x0:x1]
        scores = class_counts / class_counts.sum(axis=0)
        return scores

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Set fill_value'th class ID's count to 1 and all others to zero."""
        class_id = fill_value
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_counts[:, y0:y1, x0:x1][..., mask] = 0
        self.pixel_counts[class_id, y0:y1, x0:x1][mask] = 1

    @classmethod
    def make_empty(cls, extent: Box,
                   num_classes: int) -> 'SemanticSegmentationDiscreteLabels':
        """Instantiate an empty instance."""
        return cls(extent=extent, num_classes=num_classes)


class SemanticSegmentationSmoothLabels(SemanticSegmentationLabels):
    """Membership-scores for each pixel for each class.

    Maintains a num_classes x H x W array where value_{ijk} represents the
    probability (or some other measure) of pixel_{jk} belonging to class i.
    A discrete label array can be obtained from this by argmax'ing along the
    class dimension.
    """

    def __init__(self,
                 extent: Box,
                 num_classes: int,
                 dtype: Any = np.float16,
                 dtype_hits: Any = np.uint8):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
            dtype (Any): dtype of the scores array. Defaults to np.float16.
            dtype_hits (Any): dtype of the hits array. Defaults to np.uint8.
        """
        super().__init__(extent, num_classes, dtype)

        self.pixel_scores = np.zeros(
            (self.num_classes, self.height, self.width), dtype=self.dtype)
        self.pixel_hits = np.zeros((self.height, self.width), dtype=dtype_hits)

    def __add__(self, other: 'SemanticSegmentationSmoothLabels'
                ) -> 'SemanticSegmentationSmoothLabels':
        """Merge self with other by adding pixel scores and hits."""
        if self.extent != other.extent:
            raise ValueError('Cannot add labels with unqeual extents.')

        self.pixel_scores += other.pixel_scores
        self.pixel_hits += other.pixel_hits
        return self

    def __eq__(self, other: 'SemanticSegmentationSmoothLabels') -> bool:
        if not isinstance(other, SemanticSegmentationSmoothLabels):
            return False
        if self.extent != other.extent:
            return False
        scores_equal = np.allclose(self.pixel_scores, other.pixel_scores)
        hits_equal = np.array_equal(self.pixel_hits, other.pixel_hits)
        return (scores_equal and hits_equal)

    def __delitem__(self, window: Box) -> None:
        """Reset scores and hits to zero for pixels in the window."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        self.pixel_scores[..., y0:y1, x0:x1] = 0
        self.pixel_hits[..., y0:y1, x0:x1] = 0

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_score_arr(window)

    def add_window(self, window: Box, pixel_class_scores: np.ndarray) -> None:
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        pixel_class_scores = pixel_class_scores.astype(self.dtype)
        self.pixel_scores[..., y0:y1, x0:x1] += pixel_class_scores[..., :h, :w]
        self.pixel_hits[y0:y1, x0:x1] += 1

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get array of pixel scores."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        scores = self.pixel_scores[..., y0:y1, x0:x1]
        hits = self.pixel_hits[y0:y1, x0:x1]
        avg_scores = scores / hits
        return avg_scores

    def get_label_arr(self, window: Box,
                      null_class_id: int = -1) -> np.ndarray:
        """Get labels as array of class IDs.

        Returns null_class_id for pixels for which there is no data.
        """
        avg_scores = self.get_score_arr(window)
        label_arr = np.argmax(avg_scores, axis=0)
        mask = np.isnan(avg_scores[0])
        return np.where(mask, null_class_id, label_arr)

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Set fill_value'th class ID's score to 1 and all others to zero."""
        class_id = fill_value
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_scores[..., y0:y1, x0:x1][..., mask] = 0
        self.pixel_scores[class_id, y0:y1, x0:x1][mask] = 1
        self.pixel_hits[y0:y1, x0:x1][mask] = 1

    @classmethod
    def make_empty(cls, extent: Box,
                   num_classes: int) -> 'SemanticSegmentationSmoothLabels':
        """Instantiate an empty instance."""
        return cls(extent=extent, num_classes=num_classes)
