from rastervision.data.label import Labels
from rastervision.core.box import Box

import numpy as np
from rasterio.features import rasterize


class SemanticSegmentationLabels(Labels):
    """A set of spatially referenced labels.
    """

    def __init__(self, label_pairs=None):
        """Constructor

        Args:
            label_pairs: list of (window, label_array) where window is Box and
                label_array is numpy array of shape [height, width] and each value
                is a class_id
        """

        self.label_pairs = []
        if label_pairs is not None:
            self.label_pairs = label_pairs

    def __eq__(self, other):
        return (isinstance(other, SemanticSegmentationLabels)
                and np.array_equal(self.to_array(), other.to_array()))

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        return SemanticSegmentationLabels(self.label_pairs + other.label_pairs)

    def filter_by_aoi(self, aoi_polygons):
        """Returns a copy of these labels filtered by a given set of AOI polygons

        Converts values that lie outside of aoi_polygons to 0 (ie. don't care class)

        Args:
          aoi_polygons - A list of AOI polygons to filter by, in pixel coordinates.
        """
        arr = self.to_array()
        mask = rasterize(
            [(p, 1) for p in aoi_polygons], out_shape=arr.shape, fill=0)
        arr = arr * mask
        return SemanticSegmentationLabels.from_array(arr)

    def add_label_pair(self, window, label_array):
        self.label_pairs.append((window, label_array))

    def get_label_pairs(self):
        return self.label_pairs

    def get_extent(self):
        windows = list(map(lambda pair: pair[0], self.get_label_pairs()))
        xmax = max(map(lambda w: w.xmax, windows))
        ymax = max(map(lambda w: w.ymax, windows))
        return Box(0, 0, ymax, xmax)

    def to_array(self):
        extent = self.get_extent()
        arr = np.zeros((extent.ymax, extent.ymax))
        for window, label_array in self.label_pairs:
            arr[window.ymin:window.ymax, window.xmin:window.xmax] = label_array
        return arr

    @staticmethod
    def from_array(arr):
        window = Box(0, 0, arr.shape[0], arr.shape[1])
        label_pair = (window, arr)
        return SemanticSegmentationLabels([label_pair])

    def get_clipped_labels(self, extent):
        clipped_labels = SemanticSegmentationLabels()

        for window, label_array in self.get_label_pairs():
            clipped_window = window.intersection(extent)
            clipped_label_array = label_array[0:(
                clipped_window.ymax - clipped_window.ymin), 0:(
                    clipped_window.xmax - clipped_window.xmin)]
            clipped_labels.add_label_pair(clipped_window, clipped_label_array)

        return clipped_labels
