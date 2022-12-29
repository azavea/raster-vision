import numpy as np

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer


class NanTransformer(RasterTransformer):
    """Removes NaN values from float raster."""

    def __init__(self, to_value: float = 0.0):
        """Construct a new NanTransformer.

        Args:
            to_value: (float) NaN values are replaced
                with this
        """
        self.to_value = to_value

    def transform(self, chip, channel_order=None):
        """Transform a chip.

        Removes NaN values.

        Args:
            chip: ndarray of shape [height, width, channels] This is assumed to already
                have the channel_order applied to it if channel_order is set. In other
                words, channels should be equal to len(channel_order).

        Returns:
            [height, width, channels] numpy array

        """
        chip[np.isnan(chip)] = self.to_value
        return chip
