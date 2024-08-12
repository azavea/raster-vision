from typing import TYPE_CHECKING, Sequence

import numpy as np

from rastervision.core.data.raster_transformer import RasterTransformer
from rastervision.core.raster_stats import RasterStats
from rastervision.pipeline.utils import repr_with_args

if TYPE_CHECKING:
    from typing import Self
    from rastervision.core.data import RasterSource


class StatsTransformer(RasterTransformer):
    """Transforms non-uint8 to uint8 values using channel statistics.

    This works as follows:

    - Convert pixel values to z-scores using channel means and standard
      deviations.
    - Clip z-scores to the specified number of standard deviations (default 3)
      on each side.
    - Scale values to 0-255 and cast to uint8.

    This transformation is not applied to NODATA pixels (assumed to be pixels
    with all values equal to zero).
    """

    def __init__(self,
                 means: Sequence[float],
                 stds: Sequence[float],
                 max_stds: float = 3.):
        """Construct a new StatsTransformer.

        Args:
            means: Channel means.
            means: Channel standard deviations.
            max_stds: Number of standard deviations to clip the distribution to
                on both sides. Defaults to 3.
        """
        # shape = (1, 1, num_channels)
        self.means = np.array(means, dtype=float)
        self.stds = np.array(stds, dtype=float)
        self.max_stds = max_stds

    def transform(self, chip: np.ndarray) -> np.ndarray:
        """Clip values to +-max_stds std devs and convert to uint8 (0-255).

        Args:
            chip: Array of shape (..., H, W, C).

        Returns:
            Array of shape (..., H, W, C)
        """
        if chip.dtype == np.uint8:
            return chip

        means = self.means
        stds = self.stds
        max_stds = self.max_stds

        # Don't transform NODATA zero values.
        nodata_mask = chip == 0

        # Subtract mean and divide by std to get zscores.
        chip = chip.astype(float)
        chip -= means
        chip /= stds

        # Make zscores that fall between -max_stds and max_stds span 0 to 255.
        # range: (-max_stds, max_stds)
        chip = np.clip(chip, -max_stds, max_stds, out=chip)
        # range: [0, 2 * max_stds]
        chip += max_stds
        # range: [0, 1]
        chip /= (2 * max_stds)
        # range: [0, 255]
        chip *= 255
        chip = chip.astype(np.uint8)

        chip[nodata_mask] = 0

        return chip

    @classmethod
    def from_raster_sources(cls,
                            raster_sources: list['RasterSource'],
                            sample_prob: float | None = 0.1,
                            max_stds: float = 3.,
                            chip_sz: int = 300) -> 'Self':
        """Build with stats from the given raster sources.

        Args:
            raster_sources (list[RasterSource]): List of raster sources to
                compute stats from.
            sample_prob (float): Fraction of each raster to sample
                for computing stats. For details see docs for
                RasterStats.compute(). Defaults to 0.1.
            max_stds (float): Number of standard deviations to clip
                the distribution to on both sides. Defaults to 3.

        Returns:
            StatsTransformer: A StatsTransformer.
        """
        stats = RasterStats()
        stats.compute(
            raster_sources=raster_sources,
            sample_prob=sample_prob,
            chip_sz=chip_sz)
        stats_transformer = StatsTransformer.from_raster_stats(
            stats, max_stds=max_stds)
        return stats_transformer

    @classmethod
    def from_stats_json(cls,
                        uri: str,
                        channel_order: list[int] | None = None,
                        **kwargs) -> 'Self':
        """Build with stats from a JSON file.

        The file is expected to be in the same format as written by
        :meth:`.RasterStats.save`.

        Args:
            uri: URI of the JSON file.
            channel_order: Channel order to apply to the means and stds in the
                file.
            **kwargs: Extra args for :meth:`.__init__`.

        Returns:
            A StatsTransformer.
        """
        stats = RasterStats.load(uri)
        stats_transformer = StatsTransformer.from_raster_stats(
            stats, channel_order=channel_order, **kwargs)
        return stats_transformer

    @classmethod
    def from_raster_stats(cls,
                          stats: RasterStats,
                          channel_order: list[int] | None = None,
                          **kwargs) -> 'Self':
        """Build with stats from a :class:`.RasterStats` instance.

        The file is expected to be in the same format as written by
        :meth:`.RasterStats.save`.

        Args:
            stats: A :class:`.RasterStats` instance with non-None stats.
            channel_order: Channel order to apply to the means and stds in the
                :class:`.RasterStats`.
            **kwargs: Extra args for :meth:`.__init__`.

        Returns:
            A StatsTransformer.
        """
        means, stds = stats.means, stats.stds
        if channel_order is not None:
            means = means[channel_order]
            stds = stds[channel_order]
        stats_transformer = StatsTransformer(means, stds, **kwargs)
        return stats_transformer

    @property
    def stats(self) -> RasterStats:
        """Current statistics as a :class:`.RasterStats` instance."""
        return RasterStats(self.means, self.stds)

    def __repr__(self) -> str:
        return repr_with_args(
            self, means=self.means, stds=self.stds, max_stds=self.max_stds)

    def get_out_dtype(self, in_dtype: np.dtype) -> np.dtype:
        return np.dtype(np.uint8)
