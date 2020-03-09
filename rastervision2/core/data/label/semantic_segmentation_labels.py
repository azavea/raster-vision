from rastervision2.core.data.label import Labels

import numpy as np
from rasterio.features import rasterize
from shapely.ops import transform

from rastervision2.core.box import Box


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced semantic segmentation labels.
    """

    def __init__(self):
        self.window_to_label_arr = {}

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        self.window_to_label_arr.update(other.window_to_label_arr)
        return self

    def __eq__(self, other):
        if set(self.get_windows()) != set(other.get_windows()):
            return False

        for w in self.get_windows():
            if not self.get_label_arr(w).equal(other.get_label_arr):
                return False

        return True

    def get_windows(self):
        return [Box.from_tuple(w) for w in self.window_to_label_arr.keys()]

    def set_label_arr(self, window, label_arr):
        self.window_to_label_arr[window.tuple_format()] = label_arr

    def get_label_arr(self, window):
        return self.window_to_label_arr[window.tuple_format()]

    def filter_by_aoi(self, aoi_polygons):
        """Get the label array for a window.

        Args:
            window: Box
            extent: a Box representing the extent of the corresponding Scene

        Returns:
            np.ndarray of class_ids with zeros filled in outside the AOIs and
            clipped to the clip_extent
        """
        new_labels = SemanticSegmentationLabels()

        for window in self.get_windows():
            window_geom = window.to_shapely()
            label_arr = self.get_label_arr(window)

            if not self.aoi_polygons:
                return self
            else:
                # For each aoi_polygon, intersect with window, and put in window frame of
                # reference.
                window_aois = []
                for aoi in self.aoi_polygons:
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
                    label_arr[mask.astype(np.bool)] = 0
                    new_labels.set_label_arr(window, label_arr)

        return new_labels
