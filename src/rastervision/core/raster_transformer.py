import numpy as np

from rastervision.core.raster_stats import RasterStats


class RasterTransformer(object):
    """Transforms chips according to a config."""

    def __init__(self, options):
        """Construct a new RasterTransformer.

        Args:
            options: protos.raster_transformer_pb2.RasterTransformer
        """
        self.options = options
        self.raster_stats = None
        if options.stats_uri:
            self.raster_stats = RasterStats()
            self.raster_stats.load(options.stats_uri)

    def transform(self, chip):
        """Transform a chip.

        Selects a subset of the channels and transforms non-uint8 to
        uint8 values using options.stats_uri

        Args:
            chip: [height, width, channels] numpy array

        Returns:
            [height, width, channels] uint8 numpy array where channels is equal
                to len(self.options.channel_order)
        """
        if chip.dtype != np.uint8:
            if self.raster_stats:
                # Subtract mean and divide by std to get zscores.
                means = np.array(self.raster_stats.means)
                means = means[np.newaxis, np.newaxis, :].astype(np.float)
                stds = np.array(self.raster_stats.stds)
                stds = stds[np.newaxis, np.newaxis, :].astype(np.float)

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
                    'Need to provide stats_uri for non-uint8 rasters.')

        if self.options.channel_order:
            return chip[:, :, self.options.channel_order]
        return chip
