from rastervision.builders import raster_source_builder
from rastervision.commands.compute_stats import ComputeStats
from rastervision.utils import files
from rastervision.protos.compute_stats_pb2 import (ComputeStatsConfig)


def build(config):
    if isinstance(config, str):
        config = files.load_json_config(config, ComputeStatsConfig())

    raster_sources = [
        raster_source_builder.build(raster_source_config)
        for raster_source_config in config.raster_sources
    ]

    return ComputeStats(raster_sources, config.stats_uri)
