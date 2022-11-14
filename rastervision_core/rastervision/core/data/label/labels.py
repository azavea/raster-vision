"""Defines the abstract Labels class."""

from typing import TYPE_CHECKING, Any, Iterable
from abc import (ABC, abstractclassmethod, abstractmethod)

if TYPE_CHECKING:
    from rastervision.core.box import Box


class Labels(ABC):
    """A source-agnostic, in-memory representation of labels in a scene.

    This class can represent labels obtained via a ``LabelSource``, a
    ``LabelStore``, or directly from model predictions.
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

    @abstractmethod
    def __setitem__(self, key, value):
        pass

    @abstractclassmethod
    def make_empty(cls) -> 'Labels':
        """Instantiate an empty instance of this class.

        Returns:
            Labels: An object of the Label subclass on which this method is
                called.
        """
        pass

    @classmethod
    def from_predictions(cls, windows: Iterable['Box'],
                         predictions: Iterable[Any]) -> 'Labels':
        """Instantiate from windows and their corresponding predictions.

        This makes no assumptions about the type or format of the predictions.
        Subclasses should implement the __setitem__ method to correctly handle
        the predictions.

        Args:
            windows (Iterable[Box]): Boxes in pixel coords, specifying chips
                in the raster.
            predictions (Iterable[Any]): The model predictions for each chip
                specified by the windows.

        Returns:
            Labels: An object of the Label subclass on which this method is
                called.
        """
        labels = cls.make_empty()
        # If predictions is tqdm-wrapped, it needs to be the first arg to zip()
        # or the progress bar won't terminate with the correct count.
        for prediction, window in zip(predictions, windows):
            labels[window] = prediction
        return labels

    @abstractmethod
    def save(self, uri: str) -> None:
        """Save to file."""
        pass
