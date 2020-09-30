import re

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer

import numpy as np  # noqa


class CastTransformer(RasterTransformer):
    """Removes Cast values from float raster
    """

    def __init__(self, to_dtype: str = 'np.uint8'):
        """Construct a new CastTransformer.

        Args:
            to_dtype: (str) Chips are casted to this dtype
        """
        mo = re.search(r'np\.(u|)(int|float)[0-9]+', to_dtype)
        if mo:
            self.to_dtype = eval(mo.group(0))
        else:
            raise ValueError(f'Unsupported to_dtype {to_dtype}')

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
