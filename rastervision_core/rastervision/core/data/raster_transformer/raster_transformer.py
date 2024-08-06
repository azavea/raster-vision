from typing import TYPE_CHECKING
from abc import (ABC, abstractmethod)

if TYPE_CHECKING:
    import numpy as np


class RasterTransformer(ABC):
    """Transforms raw chips to be input to a neural network."""

    @abstractmethod
    def transform(self, chip: 'np.ndarray') -> 'np.ndarray':
        """Transform a chip of a raster source.

        Args:
            chip: Array of shape (..., H, W, C).

        Returns:
            Array of shape (..., H, W, C)
        """
