from rastervision.data.label import Labels
from rastervision.core.box import Box


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

    def __add__(self, other):
        """Add labels to these labels.

        Returns a concatenation of this and the other labels.
        """
        return SemanticSegmentationLabels(self.label_pairs + other.label_pairs)

    def filter_by_aoi(self, aoi_polygons):
        """Returns a copy of these labels filtered by a given set of AOI polygons

        Args:
          aoi_polygons - A list of AOI polygons to filter by, in pixel coordinates.
        """
        # TODO implement this
        pass

    def add_label_pair(self, window, label_array):
        self.label_pairs.append((window, label_array))

    def get_label_pairs(self):
        return self.label_pairs

    def get_extent(self):
        windows = list(map(lambda pair: pair[0], self.get_label_pairs()))
        xmax = max(map(lambda w: w.xmax, windows))
        ymax = max(map(lambda w: w.ymax, windows))
        return Box(0, 0, ymax, xmax)

    def get_clipped_labels(self, extent):
        clipped_labels = SemanticSegmentationLabels()

        for window, label_array in self.get_label_pairs():
            clipped_window = window.intersection(extent)
            clipped_label_array = label_array[0:(
                clipped_window.ymax - clipped_window.ymin), 0:(
                    clipped_window.xmax - clipped_window.xmin)]
            clipped_labels.add_label_pair(clipped_window, clipped_label_array)

        return clipped_labels
