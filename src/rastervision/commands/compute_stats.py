from rastervision.core.command import Command
from rastervision.core.raster_stats import RasterStats


class ComputeStats(Command):
    def __init__(self, raster_sources, stats_uri):
        self.raster_sources = raster_sources
        self.stats_uri = stats_uri

    def get_inputs(self):
        return [uri
                for source in self.raster_sources
                for uri in source.get_uris()]

    def get_outputs(self):
        return self.stats_uri

    def run(self):
        stats = RasterStats()
        stats.compute(self.raster_sources)
        stats.save(self.stats_uri)
