from rastervision.analyzer import Analyzer
from rastervision.core import RasterStats


class StatsAnalyzer(Analyzer):
    """Computes RasterStats against the entire scene set.
    """

    def __init__(self, stats_uri):
        self.stats_uri = stats_uri

    def process(self, scenes, tmp_dir):
        stats = RasterStats()
        stats.compute(list(map(lambda s: s.raster_source, scenes)))
        stats.save(self.stats_uri)
