from typing import TYPE_CHECKING, Iterable, Optional, Tuple
from os.path import join

from rastervision.pipeline.config import register_config, ConfigError, Field
from rastervision.core.analyzer import AnalyzerConfig, StatsAnalyzer

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig


@register_config('stats_analyzer')
class StatsAnalyzerConfig(AnalyzerConfig):
    """Configure a :class:`.StatsAnalyzer`.

    A :class:`.StatsAnalyzer` computes imagery statistics of scenes which can
    be used to normalize chips read from them.
    """

    output_uri: Optional[str] = Field(
        None,
        description='URI of directory where stats will be saved. '
        'Stats for a scene-group will be save in a JSON file at '
        '<output_uri>/<scene-group-name>/stats.json. If None, and this Config '
        'is part of an RVPipeline, this field will be auto-generated.')
    sample_prob: Optional[float] = Field(
        0.1,
        description=(
            'The probability of using a random window for computing statistics. '
            'If None, will use a sliding window.'))
    chip_sz: int = Field(
        300,
        description='Chip size to use when sampling chips to compute stats '
        'from.')
    nodata_value: Optional[float] = Field(
        0,
        description='NODATA value. If set, these pixels will be ignored when '
        'computing stats.')

    def update(self, pipeline: Optional['RVPipelineConfig'] = None) -> None:
        if pipeline is not None and self.output_uri is None:
            self.output_uri = join(pipeline.analyze_uri, 'stats')

    def validate_config(self):
        if self.sample_prob > 1 or self.sample_prob <= 0:
            raise ConfigError('sample_prob must be <= 1 and > 0')

    def build(self, scene_group: Optional[Tuple[str, Iterable[str]]] = None
              ) -> StatsAnalyzer:
        if scene_group is None:
            output_uri = join(self.output_uri, f'stats.json')
        else:
            group_name, _ = scene_group
            output_uri = join(self.output_uri, group_name, f'stats.json')
        return StatsAnalyzer(
            output_uri,
            sample_prob=self.sample_prob,
            chip_sz=self.chip_sz,
            nodata_value=self.nodata_value)

    def get_bundle_filenames(self):
        return ['stats.json']
