from abc import (ABC, abstractmethod)


class Labels(ABC):
    """A set of spatially referenced labels.

    A set of labels predicted by a model or provided by human labelers for the
    sake of training.
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

    @abstractmethod
    def __eq__(self, other):
        pass
