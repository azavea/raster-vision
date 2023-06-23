from typing import Iterable, Optional

from rastervision.core.analyzer import Analyzer
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import Scene


class StatsAnalyzer(Analyzer):
    """Compute imagery statistics of scenes."""

    def __init__(self,
                 stats_uri: Optional[str] = None,
                 sample_prob: float = 0.1,
                 chip_sz: int = 300,
                 nodata_value: Optional[float] = 0):
        self.stats_uri = stats_uri
        self.sample_prob = sample_prob
        self.chip_sz = chip_sz
        self.nodata_value = nodata_value

    def compute_stats(self, scenes: Iterable[Scene]) -> RasterStats:
        stats = RasterStats()
        stats.compute(
            [s.raster_source for s in scenes],
            sample_prob=self.sample_prob,
            chip_sz=self.chip_sz,
            nodata_value=self.nodata_value)
        return stats

    def process(self, scenes: Iterable[Scene], tmp_dir: str) -> None:
        stats = self.compute_stats(scenes)
        if self.stats_uri is not None:
            stats.save(self.stats_uri)
        else:
            raise ValueError('Cannot save stats because stats_uri is not set.')
