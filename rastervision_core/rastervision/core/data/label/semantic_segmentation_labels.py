from abc import abstractmethod
from typing import Tuple, Optional, List, Any

from rastervision.core.data.label import Labels

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision.core.box import Box


class SemanticSegmentationLabels(Labels):
    """Representation of Semantic Segmentation labels."""

    @abstractmethod
    def __add__(self, other) -> 'SemanticSegmentationLabels':
        """Merge self with other labels."""
        pass

    @abstractmethod
    def __eq__(self, other: 'SemanticSegmentationLabels') -> bool:
        pass

    @abstractmethod
    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        """Set labels for the given window, overriding current values, if any.
        """
        pass

    @abstractmethod
    def __delitem__(self, window: Box) -> None:
        """Delete labels for the given window."""
        pass

    @abstractmethod
    def __getitem__(self, window: Box) -> np.ndarray:
        """Get labels for the given window."""
        pass

    @abstractmethod
    def get_windows(self, **kwargs) -> List[Box]:
        """Get windows, optionally parameterized by keyword args."""
        pass

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
            mask = mask.astype(np.bool)
            self.mask_fill(window, mask, null_class_id)
        else:
            del self[window]

    @classmethod
    def build(self,
              smooth: bool = False,
              extent: Optional[Box] = None,
              num_classes: Optional[int] = None):
        """Constructor.

        Args:
            smooth (bool, optional): If True, creates a
                SemanticSegmentationSmoothLabels object. If False, creates a
                SemanticSegmentationDiscreteLabels object.
            extent (Optional[Box], optional): The extent of the region to which
                the labels belong, in global coordinates. Only used if
                smooth=True. Defaults to None.
            num_classes (Optional[int], optional): Number of classes.
                Only used if smooth=True. Defaults to None.

        Raises:
            ValueError: if num_classes and extent are not specified, but
                smooth=True.
        """
        if not smooth:
            return SemanticSegmentationDiscreteLabels()
        else:
            if extent is None:
                raise ValueError('extent must be specified if smooth=True.')
            if num_classes is None:
                raise ValueError(
                    'num_classes must be specified if smooth=True.')
            return SemanticSegmentationSmoothLabels(
                extent=extent, num_classes=num_classes)


class SemanticSegmentationDiscreteLabels(SemanticSegmentationLabels):
    def __init__(self):
        self.window_to_label_arr = {}

    def __add__(self, other) -> 'SemanticSegmentationDiscreteLabels':
        """Merge self with other labels by importing all window-array pairs
        into self.

        Note: This will overwrite self's values for any windows that are common
        to both self and other.
        """
        self.window_to_label_arr.update(other.window_to_label_arr)
        return self

    def __eq__(self, other: 'SemanticSegmentationDiscreteLabels') -> bool:
        # check if windows are same
        self_windows = set(self.window_to_label_arr.keys())
        other_windows = set(other.window_to_label_arr.keys())
        if self_windows != other_windows:
            return False
        # check if windows values are same
        for w in self_windows:
            arr1 = self.get_label_arr(w)
            arr2 = other.get_label_arr(w)
            if not np.array_equal(arr1, arr2):
                return False
        return True

    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        self.window_to_label_arr[window] = values

    def __delitem__(self, window: Box) -> None:
        del self.window_to_label_arr[window]

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_label_arr(window)

    def get_windows(self, **kwargs) -> List[Box]:
        return list(self.window_to_label_arr.keys())

    def get_label_arr(self, window: Box) -> np.ndarray:
        return self.window_to_label_arr[window]

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        self.window_to_label_arr[window][mask] = fill_value


class SemanticSegmentationSmoothLabels(SemanticSegmentationLabels):
    def __init__(self, extent: Box, num_classes: int):
        """Constructor.

        Args:
            extent (Box): The extent of the region to which
                the labels belong, in global coordinates.
            num_classes (int): Number of classes.
        """
        self.local_extent = extent
        self.num_classes = num_classes
        self.ymin, self.xmin, _, _ = extent
        self.height, self.width = extent.size

        # store as float16 instead of float32 to save memory
        self.dtype = np.float16

        self.pixel_scores = np.zeros(
            (num_classes, self.height, self.width), dtype=self.dtype)
        self.pixel_hits = np.zeros((self.height, self.width), dtype=np.uint8)

    def _to_local_coords(self, window: Box) -> Tuple[int, int, int, int]:
        """Convert to coordinates of the local arrays."""
        ymin, xmin, ymax, xmax = window
        ymax = min(ymax, self.height)
        xmax = min(xmax, self.width)
        return (ymin - self.ymin, xmin - self.xmin, ymax, xmax)

    def __add__(self, other) -> 'SemanticSegmentationSmoothLabels':
        """Merge self with other labels by adding the pixel scores and hits.
        """
        extents_equal = self.local_extent == other.local_extent
        coords_equal = (self.ymin, self.xmin) == (other.ymin, other.xmin)
        if not (extents_equal and coords_equal):
            raise ValueError()

        self.pixel_scores += other.pixel_scores
        self.pixel_hits += other.pixel_hits
        return self

    def __eq__(self, other: 'SemanticSegmentationSmoothLabels') -> bool:
        extents_equal = self.local_extent == other.local_extent
        coords_equal = (self.ymin, self.xmin) == (other.ymin, other.xmin)
        if not (extents_equal and coords_equal):
            return False
        scores_equal = np.allclose(self.pixel_scores, other.pixel_scores)
        hits_equal = np.array_equal(self.pixel_hits, other.pixel_hits)
        return (scores_equal and hits_equal)

    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        self.add_window(window, values)

    def __delitem__(self, window: Box) -> None:
        """Reset scores and hits for pixels in the window."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        self.pixel_scores[..., y0:y1, x0:x1] = 0
        self.pixel_hits[..., y0:y1, x0:x1] = 0

    def __getitem__(self, window: Box) -> np.ndarray:
        return self.get_score_arr(window)

    def get_windows(self, **kwargs) -> List[Box]:
        """Generate sliding windows over the local extent. The keyword args
        are passed to Box.get_windows() and can therefore be used to control
        the specifications of the windows.

        If the keyword args do not contain chip_sz, a list of length 1,
        containing the full extent is returned.
        """
        chip_sz: Optional[int] = kwargs.pop('chip_sz', None)
        if chip_sz is None:
            return [self.local_extent]
        return self.local_extent.get_windows(chip_sz, chip_sz, **kwargs)

    def add_window(self, window: Box, values: np.ndarray) -> None:
        values = values.astype(self.dtype)
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        self.pixel_scores[..., y0:y1, x0:x1] += values[..., :h, :w]
        self.pixel_hits[y0:y1, x0:x1] += 1

    def get_score_arr(self, window: Box) -> np.ndarray:
        """Get scores.

        Note: The output array is not guaranteed to be the same size as the
        input window.
        """
        y0, x0, y1, x1 = self._to_local_coords(window)
        scores = self.pixel_scores[..., y0:y1, x0:x1]
        hits = self.pixel_hits[y0:y1, x0:x1]
        avg_scores = scores / hits
        return avg_scores

    def get_label_arr(self, window: Box) -> np.ndarray:
        """Get discrete labels by argmax'ing the scoress."""
        y0, x0, y1, x1 = self._to_local_coords(window)
        avg_scores = self.get_score_arr(window)
        labels = np.argmax(avg_scores, axis=0)
        return labels

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        """Treat fill_value as a class id. Set that class's score to 1 and
        all others to zero.
        """
        class_id = fill_value
        y0, x0, y1, x1 = self._to_local_coords(window)
        h, w = y1 - y0, x1 - x0
        mask = mask[:h, :w]
        self.pixel_scores[..., y0:y1, x0:x1][..., mask] = 0
        self.pixel_scores[class_id, y0:y1, x0:x1][mask] = 1
        self.pixel_hits[y0:y1, x0:x1][mask] = 1
