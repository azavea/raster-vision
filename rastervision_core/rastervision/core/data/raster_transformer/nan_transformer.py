import numpy as np

from rastervision.core.data.raster_transformer.raster_transformer \
    import RasterTransformer


class NanTransformer(RasterTransformer):
    """Removes NaN values from float raster."""

    def __init__(self, to_value: float = 0.0):
        """Constructor.

        Args:
            to_value: NaN values are replaced with this.
        """
        self.to_value = to_value

    def transform(self, chip):
        """Removes NaN values.

        Args:
            chip: Array of shape (..., H, W, C).

        Returns:
            Array of shape (..., H, W, C)
        """
        nan_mask = np.isnan(chip)
        chip[nan_mask] = self.to_value
        return chip
