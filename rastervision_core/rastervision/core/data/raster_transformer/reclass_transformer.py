from typing import TYPE_CHECKING
from rastervision.core.data.raster_transformer import RasterTransformer

if TYPE_CHECKING:
    import numpy as np


class ReclassTransformer(RasterTransformer):
    """Maps class IDs in a label raster to other values."""

    def __init__(self, mapping: dict[int, int]):
        """Constructor.

        Args:
            mapping: Remapping dictionary, value_from-->value_to.
        """
        self.mapping = mapping

    def transform(self, chip: 'np.ndarray'):
        """Reclassify a label raster using the given mapping.

        Args:
            chip: Array of shape (..., H, W, C).

        Returns:
            Array of shape (..., H, W, C)
        """
        masks = []
        for (value_from, value_to) in self.mapping.items():
            mask = (chip == value_from)
            masks.append((mask, value_to))
        for (mask, value_to) in masks:
            chip[mask] = value_to

        return chip
