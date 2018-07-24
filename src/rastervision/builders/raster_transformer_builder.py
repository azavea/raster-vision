from rastervision.core.raster_transformer import RasterTransformer
from rastervision.core.raster_stats import RasterStats


def build(config):
    raster_stats = None
    if config.stats_uri:
        raster_stats = RasterStats()
        raster_stats.load(config.stats_uri)

    return RasterTransformer(
        channel_order=config.channel_order, raster_stats=raster_stats)
