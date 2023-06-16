from typing import TYPE_CHECKING
from abc import (ABC, abstractmethod)

if TYPE_CHECKING:
    import numpy as np


class RasterTransformer(ABC):
    """Transforms raw chips to be input to a neural network."""

    @abstractmethod
    def transform(self, chip: 'np.ndarray',
                  channel_order=None) -> 'np.ndarray':
        """Transform a chip of a raster source.

        Args:
            chip: ndarray of shape [height, width, channels] This is assumed to already
                have the channel_order applied to it if channel_order is set. In other
                words, channels should be equal to len(channel_order).
            channel_order: list of indices of channels that were extracted from the
                raw imagery.

        Returns:
            (np.ndarray): Array of shape (..., H, W, C)
        """
        pass
