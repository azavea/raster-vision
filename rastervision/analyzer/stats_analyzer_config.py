import os
from copy import deepcopy

import rastervision as rv
from rastervision.analyzer import (AnalyzerConfig, AnalyzerConfigBuilder,
                                   StatsAnalyzer)
from rastervision.protos.analyzer_pb2 import AnalyzerConfig as AnalyzerConfigMsg


class StatsAnalyzerConfig(AnalyzerConfig):
    def __init__(self, stats_uri=None, sample_prob=None):
        super().__init__(rv.STATS_ANALYZER)
        self.stats_uri = stats_uri
        self.sample_prob = sample_prob

    def create_analyzer(self):
        if not self.stats_uri:
            raise rv.ConfigError('stats_uri is not set.')
        return StatsAnalyzer(self.stats_uri, self.sample_prob)

    def to_proto(self):
        msg = AnalyzerConfigMsg(analyzer_type=self.analyzer_type)
        if self.stats_uri:
            msg.stats_analyzer_config.stats_uri = self.stats_uri
        msg.stats_analyzer_config.sample_prob = \
            (0.0 if self.sample_prob is None else self.sample_prob)
        return msg

    def save_bundle_files(self, bundle_dir):
        if not self.stats_uri:
            raise rv.ConfigError('stat_uri is not set.')
        # Only set the basename, do not contribute file
        # as it is not and input and only an output of
        # this analyzer. The StatsTransformer will save
        # its input separately.
        base_name = os.path.basename(self.stats_uri)
        new_config = self.to_builder() \
                         .with_stats_uri(base_name) \
                         .build()
        return (new_config, [])

    def load_bundle_files(self, bundle_dir):
        if not self.stats_uri:
            raise rv.ConfigError('stat_uri is not set.')
        local_stats_uri = os.path.join(bundle_dir, self.stats_uri)
        return self.to_builder() \
                   .with_stats_uri(local_stats_uri) \
                   .build()

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        if command_type == rv.ANALYZE:
            if not self.stats_uri:
                self.stats_uri = os.path.join(experiment_config.analyze_uri,
                                              'stats.json')

    def report_io(self, command_type, io_def):
        if command_type == rv.ANALYZE:
            io_def.add_output(self.stats_uri)


class StatsAnalyzerConfigBuilder(AnalyzerConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'stats_uri': prev.stats_uri,
                'sample_prob': prev.sample_prob
            }
        super().__init__(StatsAnalyzerConfig, config)

    def from_proto(self, msg):
        stats_uri = (msg.stats_analyzer_config.stats_uri or msg.stats_uri)
        b = self.with_stats_uri(stats_uri)

        sample_prob = msg.stats_analyzer_config.sample_prob
        sample_prob = (None if sample_prob == 0 else sample_prob)
        b = b.with_sample_prob(sample_prob)
        return b

    def with_stats_uri(self, stats_uri):
        """Set the stats_uri.

            Args:
                stats_uri: URI to the stats json to use
        """
        b = deepcopy(self)
        b.config['stats_uri'] = stats_uri
        return b

    def validate(self):
        sample_prob = self.config.get('sample_prob')
        if sample_prob and (not isinstance(sample_prob, float)
                            or sample_prob >= 1.0 or sample_prob <= 0):
            raise rv.ConfigError(
                'sample_prob must be a float between 0 and 1 exclusive.')

    def with_sample_prob(self, sample_prob):
        """Set the sample_prob used to sample a subset of each scene.

        If sample_prob is set, then a subset of each scene is used to compute stats which
        speeds up the computation. Roughly speaking, if sample_prob=0.5, then half the
        pixels in the scene will be used. More precisely, the number of chips is equal to
        sample_prob * (width * height / 300^2), or 1, whichever is greater. Each chip is
        uniformly sampled from the scene with replacement. Otherwise, it uses a sliding
        window over the entire scene to compute stats.

        Args:
            sample_prob: (float or None) between 0 and 1
        """
        b = deepcopy(self)
        b.config['sample_prob'] = sample_prob
        return b
