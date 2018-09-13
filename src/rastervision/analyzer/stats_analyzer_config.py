import os
from copy import deepcopy

import rastervision as rv
from rastervision.analyzer import (AnalyzerConfig, AnalyzerConfigBuilder,
                                   StatsAnalyzer)
from rastervision.protos.analyzer_pb2 import AnalyzerConfig as AnalyzerConfigMsg


class StatsAnalyzerConfig(AnalyzerConfig):
    def __init__(self, stats_uri=None):
        super().__init__(rv.STATS_ANALYZER)
        self.stats_uri = stats_uri

    def create_analyzer(self):
        if not self.stats_uri:
            raise rv.ConfigError("stat_uri is not set.")
        return StatsAnalyzer(self.stats_uri)

    def to_proto(self):
        msg = AnalyzerConfigMsg(analyzer_type=self.analyzer_type)
        if self.stats_uri:
            msg.MergeFrom(AnalyzerConfigMsg(stats_uri=self.stats_uri))
        return msg

    def preprocess_command(self, command_type, experiment_config, context=[]):
        conf = self
        io_def = rv.core.CommandIODefinition()
        if command_type == rv.ANALYZE:
            if not self.stats_uri:
                stats_uri = os.path.join(experiment_config.analyze_uri,
                                         "stats.json")
                conf = self.to_builder() \
                           .with_stats_uri(stats_uri) \
                           .build()
            io_def.add_output(conf.stats_uri)
        return (conf, io_def)


class StatsAnalyzerConfigBuilder(AnalyzerConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {"stats_uri": prev.stats_uri}
        super().__init__(StatsAnalyzerConfig, config)

    def from_proto(self, msg):
        b = StatsAnalyzerConfigBuilder()
        return b.with_stats_uri(msg.stats_uri)

    def with_stats_uri(self, stats_uri):
        """Set the stats_uri.

            Args:
                stats_uri: URI to the stats json to use
        """
        b = deepcopy(self)
        b.config['stats_uri'] = stats_uri
        return b
