from typing import Dict
from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer

import numpy as np  # noqa


class ReclassTransformer(RasterTransformer):
    """Reclassifies label raster
    """

    def __init__(self, mapping: Dict[int, int]):
        """Construct a new ReclassTransformer.

        Args:
            mapping: (dict) Remapping dictionary
        """
        self.mapping = mapping

    def transform(self, chip, channel_order=None):
        """Transform a chip.

        Reclassify a label raster using the given mapping.

        Args:
            chip: ndarray of shape [height, width, channels] This is assumed to already
                have the channel_order applied to it if channel_order is set. In other
                words, channels should be equal to len(channel_order).

        Returns:
            [height, width, channels] numpy array

        """
        masks = []
        for (value_from, value_to) in self.mapping.items():
            mask = (chip == value_from)
            masks.append((mask, value_to))
        for (mask, value_to) in masks:
            chip[mask] = value_to

        return chip
