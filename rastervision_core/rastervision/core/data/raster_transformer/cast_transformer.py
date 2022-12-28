from typing import Optional

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer

import numpy as np


class CastTransformer(RasterTransformer):
    """Casts chips to the specified dtype."""

    def __init__(self, to_dtype: str):
        """Constructor.

        Args:
            to_dtype: (str) dtype to cast the chips to.
        """
        self.to_dtype = np.dtype(to_dtype)

    def __repr__(self):
        return f'CastTransformer(to_dtype="{self.to_dtype}")'

    def transform(self, chip: np.ndarray,
                  channel_order: Optional[list] = None) -> np.ndarray:
        """Cast chip to self.to_dtype.

        Args:
            chip: ndarray of shape [height, width, channels]

        Returns:
            [height, width, channels] numpy array
        """
        return chip.astype(self.to_dtype)
