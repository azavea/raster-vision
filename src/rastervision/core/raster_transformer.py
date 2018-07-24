import numpy as np


class RasterTransformer(object):
    """Transforms raw chips to be input to a neural network."""

    def __init__(self, channel_order=None, raster_stats=None):
        """Construct a new RasterTransformer.

        Args:
            channel_order: numpy array of length n where n is the number of
                channels to use and the values are channel indices
            raster_stats: (RasterStats) used to transform chip to have
                desired statistics
        """
        self.channel_order = channel_order
        self.raster_stats = raster_stats

    def transform(self, chip):
        """Transform a chip.

        Selects a subset of the channels and transforms non-uint8 to
        uint8 values using raster_stats.

        Args:
            chip: [height, width, channels] numpy array

        Returns:
            [height, width, channels] uint8 numpy array where channels is equal
                to len(channel_order)
        """
        if self.channel_order is None:
            channel_order = np.arange(chip.shape[2])
        else:
            channel_order = self.channel_order

        chip = chip[:, :, channel_order]

        if chip.dtype != np.uint8:
            if self.raster_stats:
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
                raise ValueError(
                    'Need to provide raster_stats for non-uint8 rasters.')

        return chip
