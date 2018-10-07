import numpy as np

from rastervision.data.raster_transformer.raster_transformer \
    import RasterTransformer


class StatsTransformer(RasterTransformer):
    """Transforms non-uint8 to uint8 values using raster_stats.
    """

    def __init__(self, raster_stats=None):
        """Construct a new StatsTransformer.

        Args:
            raster_stats: (RasterStats) used to transform chip to have
                desired statistics
        """
        self.raster_stats = raster_stats

    def transform(self, chip, channel_order=None):
        """Transform a chip.

        Transforms non-uint8 to uint8 values using raster_stats.

        Args:
            chip: [height, width, channels] numpy array
            channel_order: The channel order to use for these statistics.

        Returns:
            [height, width, channels] uint8 numpy array
        """
        if chip.dtype != np.uint8:
            if self.raster_stats:
                if channel_order is None:
                    channel_order = np.arange(chip.shape[2])

                # Subtract mean and divide by std to get zscores.
                means = np.array(self.raster_stats.means)
                means = means[np.newaxis, np.newaxis, channel_order].astype(
                    np.float)
                stds = np.array(self.raster_stats.stds)
                stds = stds[np.newaxis, np.newaxis, channel_order].astype(
                    np.float)

                # Don't transform NODATA zero values.
                nodata = chip == 0

                chip = chip - means
                chip = chip / stds

                # Make zscores that fall between -3 and 3 span 0 to 255.
                chip += 3
                chip /= 6

                chip = np.clip(chip, 0, 1)
                chip *= 255
                chip = chip.astype(np.uint8)

                chip[nodata] = 0
            else:
                raise ValueError('raster_stats not defined.')

        return chip
