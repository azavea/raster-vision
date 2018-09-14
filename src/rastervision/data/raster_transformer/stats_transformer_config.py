from copy import deepcopy

import rastervision as rv
from rastervision.core.raster_stats import RasterStats
from rastervision.data.raster_transformer import (
    RasterTransformerConfig, RasterTransformerConfigBuilder, StatsTransformer,
    NoopTransformer)
from rastervision.protos.raster_transformer2_pb2 \
    import RasterTransformerConfig as RasterTransformerConfigMsg


class StatsTransformerConfig(RasterTransformerConfig):
    def __init__(self, stats_uri=None):
        super().__init__(rv.STATS_TRANSFORMER)
        self.stats_uri = stats_uri

    def to_proto(self):
        msg = RasterTransformerConfigMsg(
            transformer_type=self.transformer_type, stats_uri=self.stats_uri)
        return msg

    def create_transformer(self):
        if not self.stats_uri:
            return NoopTransformer()

        return StatsTransformer(RasterStats.load(self.stats_uri))

    def preprocess_command(self, command_type, experiment_config, context=[]):
        conf = self
        io_def = rv.core.CommandIODefinition()
        if command_type != rv.ANALYZE:
            if not conf.stats_uri:
                # Find the stats URI from a StatsAnalyzer
                for analyzer in experiment_config.analyzers:
                    if analyzer.analyzer_type == rv.STATS_ANALYZER:
                        stats_uri = analyzer.stats_uri
                        conf = self.to_builder() \
                                   .with_stats_uri(stats_uri) \
                                   .build()
            if not conf.stats_uri:
                io_def.add_missing(
                    "StatsTransformerConfig is missing 'stats_uri' property "
                    'in command {}. '
                    'This must be set on the configuration, or a '
                    'StatsAnalyzerConfig must be added to '
                    'this experiment.'.format(command_type))
            else:
                io_def.add_input(conf.stats_uri)

        return (conf, io_def)


class StatsTransformerConfigBuilder(RasterTransformerConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {'stats_uri': prev.stats_uri}
        super().__init__(StatsTransformerConfig, config)

    def from_proto(self, msg):
        b = StatsTransformerConfigBuilder()
        return b.with_stats_uri(msg.stats_uri)

    def with_stats_uri(self, stats_uri):
        """Set the stats_uri.

            Args:
                stats_uri: URI to the stats json to use
        """
        b = deepcopy(self)
        b.config['stats_uri'] = stats_uri
        return b
