from typing import List, Optional

import numpy as np

from rastervision.core.data.raster_transformer import RasterTransformer


class MinMaxTransformer(RasterTransformer):
    """Transforms chips by scaling values in each channel to span 0-255."""

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[List[int]] = None) -> np.ndarray:
        c = chip.shape[-1]
        pixels = chip.reshape(-1, c)
        channel_mins = pixels.min(axis=0)
        channel_maxs = pixels.max(axis=0)
        chip_normalized = (chip - channel_mins) / (channel_maxs - channel_mins)
        chip_normalized = (255 * chip_normalized).astype(np.uint8)
        return chip_normalized
