from abc import (ABC, abstractmethod)


class Labels(ABC):
    """A set of spatially referenced labels.
    A set of labels predicted by a model or provided by human labelers for the
    sake of training. Every label is associated with a spatial location and a
    class. For object detection, a label is a bounding box surrounding an
    object and the associated class. For classification, a label is a bounding
    box representing a cell/chip within a spatial grid and its class.
    For segmentation, a label is a pixel and its class.
    """

    @abstractmethod
    def __add__(self, other):
        """Add labels to these labels.
        Returns a concatenation of this and the other labels.
        """
        pass

    @abstractmethod
    def filter_by_aoi(self, aoi_polygons):
        """Returns a copy of these labels filtered by a given set of AOI polygons

        Args:
          aoi_polygons - A list of AOI polygons to filter by, in pixel coordinates.
        """
        pass
