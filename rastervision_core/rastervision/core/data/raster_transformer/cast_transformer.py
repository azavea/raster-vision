
from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer

import numpy as np


class CastTransformer(RasterTransformer):
    """Removes Cast values from float raster
    """

    def __init__(self, to_dtype: str):
        """Construct a new CastTransformer.

        Args:
            to_dtype: (str) Chips are casted to this dtype
        """
        self.to_dtype = np.dtype(to_dtype)

    def transform(self, chip, channel_order=None):
        """Transform a chip.

        Cast chip to the specified dtype.

        Args:
            chip: ndarray of shape [height, width, channels] This is assumed to already
                have the channel_order applied to it if channel_order is set. In other
                words, channels should be equal to len(channel_order).

        Returns:
            [height, width, channels] numpy array

        """
        return chip.astype(self.to_dtype)
