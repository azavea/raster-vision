from rastervision.analyzer import Analyzer
from rastervision.core import RasterStats


class StatsAnalyzer(Analyzer):
    """Computes RasterStats against the entire scene set.
    """

    def __init__(self, stats_uri, sample_prob=None):
        self.stats_uri = stats_uri
        self.sample_prob = sample_prob

    def process(self, scenes, tmp_dir):
        stats = RasterStats()
        stats.compute(
            list(map(lambda s: s.raster_source, scenes)),
            sample_prob=self.sample_prob)
        stats.save(self.stats_uri)
