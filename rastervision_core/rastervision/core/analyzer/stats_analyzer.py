from typing import List

from rastervision.core.analyzer import Analyzer
from rastervision.core.raster_stats import RasterStats
from rastervision.core.data import Scene


class StatsAnalyzer(Analyzer):
    """Computes RasterStats against the entire scene set."""

    def __init__(self, stats_uri: str, sample_prob: float = 0.1):
        self.stats_uri = stats_uri
        self.sample_prob = sample_prob

    def process(self, scenes: List[Scene], tmp_dir: str):
        stats = RasterStats()
        stats.compute(
            [s.raster_source for s in scenes], sample_prob=self.sample_prob)
        stats.save(self.stats_uri)
