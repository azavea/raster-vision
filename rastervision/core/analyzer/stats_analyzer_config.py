from typing import Optional
from os.path import join

from rastervision.pipeline.config import register_config, ConfigError, Field
from rastervision.core.analyzer import AnalyzerConfig


@register_config('stats_analyzer')
class StatsAnalyzerConfig(AnalyzerConfig):
    """Config for an Analyzer that computes imagery statistics of scenes."""

    output_uri: Optional[str] = Field(
        None,
        description=(
            'URI for output. If None and this is part of an RVPipeline, this is '
            'auto-generated.'))
    sample_prob: Optional[float] = Field(
        0.1,
        description=(
            'The probability of using a random window for computing statistics. '
            'If None, will use a sliding window.'))

    def update(self, pipeline=None):
        if pipeline is not None and self.output_uri is None:
            self.output_uri = join(pipeline.analyze_uri, 'stats.json')

    def validate_config(self):
        if self.sample_prob > 1 or self.sample_prob <= 0:
            raise ConfigError('sample_prob must be <= 1 and > 0')

    def build(self):
        from rastervision.core.analyzer import StatsAnalyzer
        return StatsAnalyzer(self.output_uri, self.sample_prob)

    def get_bundle_filenames(self):
        return ['stats.json']
