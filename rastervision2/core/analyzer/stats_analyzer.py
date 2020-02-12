from rastervision2.core.analyzer import Analyzer
from rastervision2.core.raster_stats import RasterStats


class StatsAnalyzer(Analyzer):
    """Computes RasterStats against the entire scene set.
    """

    def __init__(self, stats_uri, sample_prob=0.1):
        self.stats_uri = stats_uri
        self.sample_prob = sample_prob

    def process(self, scenes, tmp_dir):
        stats = RasterStats()
        stats.compute(
            [s.raster_source for s in scenes], sample_prob=self.sample_prob)
        stats.save(self.stats_uri)
