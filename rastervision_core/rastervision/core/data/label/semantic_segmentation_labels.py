from typing import Tuple, Optional, List, Any, Dict

from rastervision.core.data.label import Labels

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision.core.box import Box


class SemanticSegmentationLabels(Labels):
    """Representation of Semantic Segmentation labels.

    These labels can be either discrete or smooth.

    If smooth, they store the scores for each class, for each pixel.
    They also store the number of "hits" (initially equal to 1) for each pixel
    that are used to divide the values for that pixel to get an average.
    This allows one to do multiple updates to the same pixel
    (the hits will be automatically incremented by 1 each time)
    and have its final value be the average of all the updates.

    If discrete, the labels are stored as a dict, mapping windows to label
    arrays.

    """

    def __init__(self,
                 smooth: bool = False,
                 extent: Optional[Box] = None,
                 num_classes: Optional[int] = None):
        """Constructor.

        Args:
            smooth (bool, optional): If True, labels are stored as continuous
                values representing class scores instead of discrete labels.
                These values will be stored as a (C, H, W) array. If False,
                the labels are stored as a mapping of windows to label arrays.
                Defaults to False.
            extent (Optional[Box], optional): The extent of the region to which
                the labels belong, in global coordinates. Only used if
                smooth=True. Defaults to None.
            num_classes (Optional[int], optional): Number of classes.
                Only used if smooth=True. Defaults to None.

        Raises:
            ValueError: [description]
        """
        self.smooth = smooth

        if not self.smooth:
            self.window_to_label_arr: Dict[Box, np.ndarray] = {}
        else:
            if extent is None:
                raise ValueError('extent must be specified if smooth=True.')
            if num_classes is None:
                raise ValueError(
                    'num_classes must be specified if smooth=True.')

            self.local_extent = extent
            self.num_classes = num_classes
            self.ymin, self.xmin, _, _ = extent
            self.height, self.width = extent.size

            # store as float16 instead of float32 to save memory
            self.dtype = np.float16

            self.pixel_scores = np.zeros(
                (num_classes, self.height, self.width), dtype=self.dtype)
            self.pixel_hits = np.zeros(
                (self.height, self.width), dtype=np.uint8)

    def _to_local_coords(self, window: Box) -> Tuple[int, int, int, int]:
        """Convert to coordinates of the local arrays."""
        ymin, xmin, ymax, xmax = window
        ymax = min(ymax, self.height)
        xmax = min(xmax, self.width)
        return (ymin - self.ymin, xmin - self.xmin, ymax, xmax)

    def __add__(self, other) -> 'SemanticSegmentationLabels':
        """Merge self with other labels.

        If not smooth, update the window-to-label mapping. This will overwrite
        self's values for any windows that are shared.

        If smooth, add the pixel scores and hits.
        """
        if not self.smooth:
            self.window_to_label_arr.update(other.window_to_label_arr)
            return self

        smooths_equal = self.smooth == other.smooth
        extents_equal = self.local_extent == other.local_extent
        coords_equal = (self.ymin, self.xmin) == (other.ymin, other.xmin)
        if not (smooths_equal and extents_equal and coords_equal):
            raise ValueError()

        self.pixel_scores += other.pixel_scores
        self.pixel_hits += other.pixel_hits
        return self

    def __eq__(self, other: 'SemanticSegmentationLabels') -> bool:
        if self.smooth == other.smooth:
            return False

        if not self.smooth:
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

        extents_equal = self.local_extent == other.local_extent
        coords_equal = (self.ymin, self.xmin) == (other.ymin, other.xmin)
        if not (extents_equal and coords_equal):
            return False
        scores_equal = np.array_equal(self.pixel_scores, other.pixel_scores)
        hits_equal = np.array_equal(self.pixel_hits, other.pixel_hits)
        return (scores_equal and hits_equal)

    def __setitem__(self, window: Box, values: np.ndarray) -> None:
        """Set labels for the given window.

        If not smooth, update the window-to-label mapping. This will overwrite
        self's values (if they have been previously set) for this window.

        If smooth, overwrite the pixel scores and reset pixel hits to 1.
        This will overwrite self's values (if they have been previously set)
        for this window.
        """
        if not self.smooth:
            self.window_to_label_arr[window] = values
        else:
            values = values.astype(self.dtype)
            y0, x0, y1, x1 = self._to_local_coords(window)
            h, w = y1 - y0, x1 - x0
            self.pixel_scores[..., y0:y1, x0:x1] = values[..., :h, :w]
            self.pixel_hits[..., y0:y1, x0:x1] = 1

    def __delitem__(self, window: Box) -> None:
        """Delete labels for the given window.

        If not smooth, delete window from dict.

        If smooth, reset pixel scores and pixel hits to 0.
        """
        if not self.smooth:
            del self.window_to_label_arr[window]
        else:
            y0, x0, y1, x1 = self._to_local_coords(window)
            self.pixel_scores[..., y0:y1, x0:x1] = 0
            self.pixel_hits[..., y0:y1, x0:x1] = 0

    def __getitem__(self, window: Box) -> np.ndarray:
        if self.smooth:
            return self.get_score_arr(window)
        return self.get_label_arr(window)

    def get_windows(self, **kwargs) -> List[Box]:
        """Get windows.

        If not smooth, return all the windows in the dict.

        If smooth, generate sliding windows over the local extent. The window
        specifications can be configured by providing chip_sz, stride, and
        padding as keyword arguments.
        """
        if not self.smooth:
            return list(self.window_to_label_arr.keys())

        chip_sz: Optional[int] = kwargs.pop('chip_sz', None)
        if chip_sz is None:
            return [self.local_extent]
        return self.local_extent.get_windows(chip_sz, chip_sz, **kwargs)

    def add_window(self, window: Box, values: np.ndarray) -> None:
        """Add a window and its values.

        If not smooth, update the window-to-label mapping. This will overwrite
        self's values for any windows that are shared.

        If smooth, add the pixel scores and hits.
        """
        if not self.smooth:
            self.window_to_label_arr[window] = values
        else:
            values = values.astype(self.dtype)
            y0, x0, y1, x1 = self._to_local_coords(window)
            h, w = y1 - y0, x1 - x0
            self.pixel_scores[..., y0:y1, x0:x1] += values[..., :h, :w]
            self.pixel_hits[y0:y1, x0:x1] += 1

    def get_score_arr(self, window: Box, null_class_id=None) -> np.ndarray:
        """Get scores.

        Note: The output array is not guaranteed to be the same size as the
        input window.
        """
        if not self.smooth:
            return NotImplementedError(
                'get_score_arr() not supported for smooth=False.')

        y0, x0, y1, x1 = self._to_local_coords(window)
        scores = self.pixel_scores[..., y0:y1, x0:x1]
        hits = self.pixel_hits[y0:y1, x0:x1]
        avg_scores = scores / hits
        return avg_scores

    def get_label_arr(self, window: Box) -> np.ndarray:
        """Get discret labels.

        If smooth, the scores are argmax'd to get discrete labels.
        """
        if not self.smooth:
            return self.window_to_label_arr[window]

        y0, x0, y1, x1 = self._to_local_coords(window)
        avg_scores = self.get_score_arr(window)
        labels = np.argmax(avg_scores, axis=0)
        return labels

    def filter_by_aoi(self, aoi_polygons: list, null_class_id: int,
                      **kwargs) -> 'SemanticSegmentationLabels':
        """Keep only the values that lie inside the AOI.
        """
        if not aoi_polygons:
            return self

        if not self.smooth:
            for window in self.get_windows(**kwargs):
                self._filter_window_by_aoi(window, aoi_polygons, null_class_id)
        else:
            self._filter_window_by_aoi(self.local_extent, aoi_polygons,
                                       null_class_id)
        return self

    def mask_fill(self, window: Box, mask: np.ndarray,
                  fill_value: Any) -> None:
        if not self.smooth:
            self.window_to_label_arr[window][mask] = fill_value
        else:
            y0, x0, y1, x1 = self._to_local_coords(window)
            h, w = y1 - y0, x1 - x0
            mask = mask[:h, :w]
            self.pixel_scores[..., y0:y1, x0:x1][..., mask] = fill_value
            self.pixel_hits[y0:y1, x0:x1][mask] = 1

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
                out_shape=label_arr.shape,
                fill=1,
                dtype=np.uint8)
            mask = mask.astype(np.bool)
            self.mask_fill(window, mask, null_class_id)
        else:
            del self[window]
