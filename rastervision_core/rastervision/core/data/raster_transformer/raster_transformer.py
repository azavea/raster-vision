from typing import TYPE_CHECKING
from abc import (ABC, abstractmethod)
from pydantic.types import PositiveInt as PosInt

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

    def get_out_channels(self, in_channels: PosInt) -> PosInt:
        """Number of channels in output of ``transform()``."""
        return in_channels

    def get_out_dtype(self, in_dtype: 'np.dtype') -> 'np.dtype':
        """dtype of the output of ``transform()``."""
        return in_dtype
