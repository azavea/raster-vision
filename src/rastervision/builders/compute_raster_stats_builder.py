from rastervision.builders import raster_source_builder
from rastervision.commands.compute_raster_stats import ComputeRasterStats
from rastervision.utils import files
from rastervision.protos.compute_raster_stats_pb2 import (
    ComputeRasterStatsConfig)


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, ComputeRasterStatsConfig())

    raster_sources = [
        raster_source_builder.build(raster_source_config)
        for raster_source_config in config.raster_sources
    ]

    return ComputeRasterStats(raster_sources, config.stats_uri)
