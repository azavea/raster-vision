from typing import (TYPE_CHECKING, Iterable, Iterator, Optional, Sequence,
                    Tuple, Union)

import numpy as np
from tqdm.auto import tqdm

from rastervision.pipeline.utils import repr_with_args
from rastervision.pipeline.file_system import file_to_json, json_to_file
from rastervision.core.data.utils import ensure_json_serializable

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import RasterSource


class RasterStats:
    """Band-wise means and standard deviations."""

    def __init__(self,
                 means: Optional[np.ndarray] = None,
                 stds: Optional[np.ndarray] = None,
                 counts: Optional[np.ndarray] = None):
        """Constructor.

        Args:
            means (Optional[np.ndarray]): Band means. Defaults to None.
            stds (Optional[np.ndarray]): Band standard deviations.
                Defaults to None.
            counts (Optional[np.ndarray]): Band pixel counts (used to compute
                the specified means and stds). Defaults to None.
        """
        self.means = means
        self.stds = stds
        self.counts = counts

    @classmethod
    def load(cls, stats_uri: str) -> 'RasterStats':
        """Load stats from file."""
        stats_json = file_to_json(stats_uri)
        assert 'means' in stats_json and 'stds' in stats_json
        stats = RasterStats(
            means=stats_json['means'],
            stds=stats_json['stds'],
            counts=stats_json.get('counts'))
        return stats

    def compute(self,
                raster_sources: Sequence['RasterSource'],
                sample_prob: Optional[float] = None,
                chip_sz: int = 300,
                stride: Optional[int] = None,
                nodata_value: Optional[float] = 0) -> None:
        """Compute the mean and stds over all the raster_sources.

        This ignores NODATA values if nodata_value is not None.

        If sample_prob is set, then a subset of each scene is used to compute
        stats which speeds up the computation. Roughly speaking, if
        sample_prob=0.5, then half the pixels in the scene will be used. More
        precisely, the number of chips is equal to
        sample_prob * (width * height / 300^2), or 1, whichever is greater.
        Each chip is uniformly sampled from the scene with replacement.
        Otherwise, it uses a sliding window over the entire scene to compute
        stats.

        Args:
            raster_sources Sequence['RasterSource']: List of RasterSources.
            sample_prob (Optional[float]): Pixel sampling probability. See
                notes above. Defaults to None.
            nodata_value (Optional[float]): NODATA value. If set, these pixels
                will be ignored when computing stats.
        """
        if sample_prob is None:
            if stride is None:
                stride = chip_sz
            chip_stream = sliding_chip_stream(
                raster_sources, chip_sz, stride, nodata_value=nodata_value)
        else:
            chip_stream = random_chip_stream(
                raster_sources,
                chip_sz,
                sample_prob,
                nodata_value=nodata_value)

        means, vars, counts = self.compute_from_chips(
            chip_stream,
            running_mean=self.means,
            running_var=self.vars,
            running_count=self.counts)
        if means is None or vars is None:
            raise ValueError('No valid chips found in raster sources to '
                             'compute stats from. This may be because all '
                             'sampled chips were entirely composed of NODATA '
                             'pixels.')
        self.means = means
        self.stds = np.sqrt(vars)
        self.counts = counts

    def compute_from_chips(
            self,
            chips: Iterable[np.ndarray],
            running_mean: Optional[np.ndarray] = None,
            running_var: Optional[np.ndarray] = None,
            running_count: Optional[np.ndarray] = None) -> Union[Tuple[
                None, None, None], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute running mean and var from chips in stream."""
        with tqdm(chips, desc='Analyzing chips') as bar:
            for chip in bar:
                num_channels = chip.shape[-1]
                # (..., H, W, C) --> (... * H * W, C)
                pixels = chip.reshape(-1, num_channels)
                stats = self.compute_from_pixels(pixels, running_mean,
                                                 running_var, running_count)
                running_mean, running_var, running_count = stats

        return running_mean, running_var, running_count

    def compute_from_pixels(self,
                            pixels: np.ndarray,
                            running_mean: Optional[np.ndarray] = None,
                            running_var: Optional[np.ndarray] = None,
                            running_count: Optional[np.ndarray] = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Update running mean and var from pixel values."""
        running_stats = [running_mean, running_var, running_count]
        has_running_stats = any(s is not None for s in running_stats)
        has_all_running_stats = all(s is not None for s in running_stats)
        if has_running_stats and not has_all_running_stats:
            raise ValueError('Provide either none or all running stats.')

        channel_means = np.nanmean(pixels, axis=0)
        channel_vars = np.nanvar(pixels, axis=0)
        channel_counts = np.sum(~np.isnan(pixels), axis=0)

        if not has_running_stats:
            return channel_means, channel_vars, channel_counts

        running_var = parallel_variance(channel_means, channel_counts,
                                        channel_vars, running_mean,
                                        running_count, running_var)
        running_mean = parallel_mean(channel_means, channel_counts,
                                     running_mean, running_count)
        running_count += channel_counts

        return running_mean, running_var, running_count

    def to_dict(self) -> dict:
        stats_dict = dict(means=self.means, stds=self.stds, counts=self.counts)
        return stats_dict

    def save(self, stats_uri: str) -> None:
        """Save stats to file."""
        assert self.means is not None and self.stds is not None
        stats_dict = self.to_dict()
        stats_dict = ensure_json_serializable(stats_dict)
        json_to_file(stats_dict, stats_uri)

    @property
    def vars(self) -> Optional[np.ndarray]:
        """Channel variances, if self.stds is set."""
        if self.stds is None:
            return None
        return self.stds**2

    def __repr__(self) -> str:
        return repr_with_args(self, **self.to_dict())


def parallel_variance(mean_a, count_a, var_a, mean_b, count_b, var_b):
    """Compute the variance based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        var_a: the variance of partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b
        var_b: the variance of partition b

    Return:
        the variance of the two partitions if they were combined
    """
    delta = mean_b - mean_a
    m_a = var_a * (count_a - 1)
    m_b = var_b * (count_b - 1)
    M2 = m_a + m_b + delta**2 * count_a * count_b / (count_a + count_b)
    var = M2 / (count_a + count_b - 1)
    return var


def parallel_mean(mean_a, count_a, mean_b, count_b):
    """Compute the mean based on stats from two partitions of the data.

    See "Parallel Algorithm" in
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    Args:
        mean_a: the mean of partition a
        count_a: the number of elements in partition a
        mean_b: the mean of partition b
        count_b: the number of elements in partition b

    Return:
        the mean of the two partitions if they were combined
    """
    mean = (count_a * mean_a + count_b * mean_b) / (count_a + count_b)
    return mean


def sliding_chip_stream(
        raster_sources: Iterable['RasterSource'],
        chip_sz: int,
        stride: int,
        nodata_value: Optional[float] = 0) -> Iterator[np.ndarray]:
    """Get stream of chips using a sliding window."""
    for raster_source in raster_sources:
        windows = raster_source.extent.get_windows(chip_sz, stride)
        for window in windows:
            chip = get_chip(raster_source, window, nodata_value=nodata_value)
            if chip is None:
                continue
            yield chip


def random_chip_stream(
        raster_sources: Iterable['RasterSource'],
        chip_sz: int,
        sample_prob: float,
        nodata_value: Optional[float] = 0) -> Iterator[np.ndarray]:
    """Get random stream of chips."""
    for raster_source in raster_sources:
        extent = raster_source.extent
        num_chips_to_sample = get_num_chips_to_sample(extent, chip_sz,
                                                      sample_prob)
        if num_chips_to_sample == 0:
            windows = [extent]
        else:
            windows = [
                extent.make_random_square(chip_sz)
                for _ in range(num_chips_to_sample)
            ]
        for window in windows:
            chip = get_chip(raster_source, window, nodata_value=nodata_value)
            if chip is None:
                continue
            yield chip


def get_chip(raster_source: 'RasterSource',
             window: 'Box',
             nodata_value: Optional[float] = 0) -> Optional[np.ndarray]:
    """Return chip or None if all values are NODATA."""
    chip = raster_source.get_raw_chip(window).astype(float)

    if nodata_value is None:
        return chip

    chip[chip == nodata_value] = np.nan
    all_nan_pixels = np.all(np.isnan(chip))
    if all_nan_pixels:
        return None
    return chip


def get_num_chips_to_sample(extent: 'Box', chip_sz: int,
                            sample_prob: float) -> int:
    num_pixels_total = extent.area
    num_pixels_per_chip = chip_sz**2
    if num_pixels_per_chip > num_pixels_total:
        return 0
    num_chips_total = (num_pixels_total / num_pixels_per_chip)
    num_chips_to_sample = round(sample_prob * num_chips_total)
    return max(1, num_chips_to_sample)
