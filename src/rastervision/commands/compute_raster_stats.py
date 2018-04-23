from rastervision.core.command import Command
from rastervision.core.raster_stats import RasterStats


class ComputeRasterStats(Command):
    def __init__(self, raster_sources, stats_uri):
        self.raster_sources = raster_sources
        self.stats_uri = stats_uri

    def run(self):
        stats = RasterStats()
        stats.compute(self.raster_sources)
        stats.save(self.stats_uri)
