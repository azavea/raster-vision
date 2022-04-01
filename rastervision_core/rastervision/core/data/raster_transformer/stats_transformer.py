from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from rastervision.core.data.raster_transformer import RasterTransformer

if TYPE_CHECKING:
    from rastervision.core.raster_stats import RasterStats


class StatsTransformer(RasterTransformer):
    """Transforms non-uint8 to uint8 values using raster_stats.
    """

    def __init__(self, raster_stats: 'RasterStats'):
        """Construct a new StatsTransformer.

        Args:
            raster_stats: (RasterStats) used to transform chip to have
                desired statistics
        """
        # shape = (1, 1, num_channels)
        self.means = np.array(
            raster_stats.means, dtype=float)[np.newaxis, np.newaxis, :]
        self.stds = np.array(
            raster_stats.stds, dtype=float)[np.newaxis, np.newaxis, :]

    def transform(self,
                  chip: np.ndarray,
                  channel_order: Optional[Sequence[int]] = None) -> np.ndarray:
        """Transform a chip.

        Transforms non-uint8 to uint8 values using raster_stats.

        Args:
            chip: ndarray of shape [height, width, channels] This is assumed to already
                have the channel_order applied to it if channel_order is set. In other
                words, channels should be equal to len(channel_order).
            channel_order: list of indices of channels that were extracted from the
                raw imagery.

        Returns:
            [height, width, channels] uint8 numpy array

        """
        if chip.dtype != np.uint8:
            if channel_order is None:
                channel_order = np.arange(chip.shape[2])

            # Don't transform NODATA zero values.
            nodata_mask = chip == 0

            # Subtract mean and divide by std to get zscores.
            chip = chip.astype(float)
            chip -= self.means[..., channel_order]
            chip /= self.stds[..., channel_order]

            # Make zscores that fall between -3 and 3 span 0 to 255.
            chip = np.clip(chip, -3, 3, out=chip)
            chip += 3
            chip /= 6
            chip *= 255
            chip = chip.astype(np.uint8)

            chip[nodata_mask] = 0

        return chip
