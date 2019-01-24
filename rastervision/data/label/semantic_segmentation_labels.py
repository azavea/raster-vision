from rastervision.data.label import Labels

import numpy as np
from rasterio.features import rasterize
import shapely


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced semantic segmentation labels.

    Since labels are represented as rasters, the labels for a scene can take up a lot of
    memory. Therefore, to avoid running out of memory, labels are computed as needed for
    windows.
    """

    def __init__(self, windows, label_fn, aoi_polygons=None):
        """Constructor

        Args:
            windows: a list of Box representing the windows covering a scene
            label_fn: a function that takes a window (Box) and returns a label array
                of the same shape with each value a class id.
            aoi_polygons: a list of shapely.geom that contains the AOIs
                (areas of interest) for a scene.

        """
        self.windows = windows
        self.label_fn = label_fn
        self.aoi_polygons = aoi_polygons

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        return SemanticSegmentationLabels(
            self.windows + other.windows,
            self.label_fn,
            aoi_polygons=self.aoi_polygons)

    def __eq__(self, other):
        for window in self.get_windows():
            if not np.array_equal(
                    self.get_label_arr(window), other.get_label_arr(window)):
                return False
        return True

    def filter_by_aoi(self, aoi_polygons):
        """Returns a new SemanticSegmentationLabels object with aoi_polygons set."""
        return SemanticSegmentationLabels(
            self.windows, self.label_fn, aoi_polygons=aoi_polygons)

    def add_window(self, window):
        self.windows.append(window)

    def get_windows(self):
        return self.windows

    def get_label_arr(self, window, clip_extent=None):
        """Get the label array for a window.

        Note: the window should be kept relatively small to avoid running out of memory.

        Args:
            window: Box
            clip_extent: a Box representing the extent of the corresponding Scene

        Returns:
            np.ndarray of class_ids with zeros filled in outside the AOIs and clipped
                to the clip_extent
        """
        window_geom = window.to_shapely()

        if not self.aoi_polygons:
            label_arr = self.label_fn(window)
        else:
            # For each aoi_polygon, intersect with window, and put in window frame of
            # reference.
            window_aois = []
            for aoi in self.aoi_polygons:
                window_aoi = aoi.intersection(window_geom)
                if not window_aoi.is_empty:

                    def transform_shape(x, y, z=None):
                        return (x - window.xmin, y - window.ymin)

                    window_aoi = shapely.ops.transform(transform_shape,
                                                       window_aoi)
                    window_aois.append(window_aoi)

            if window_aois:
                # If window intersects with AOI, set pixels outside the AOI polygon to 0,
                # so they are ignored during eval.
                label_arr = self.label_fn(window)
                mask = rasterize(
                    [(p, 0) for p in window_aois],
                    out_shape=label_arr.shape,
                    fill=1,
                    dtype=np.uint8)
                label_arr[mask.astype(np.bool)] = 0
            else:
                # If window does't overlap with any AOI, then return all zeros.
                label_arr = np.zeros((window.get_height(), window.get_width()))

        if clip_extent is not None:
            clip_window = window.intersection(clip_extent)
            label_arr = label_arr[0:clip_window.get_height(), 0:
                                  clip_window.get_width()]

        return label_arr
