from typing import TYPE_CHECKING, Optional
from os.path import join

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       StatsTransformer)
from rastervision.core.raster_stats import RasterStats

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.core.data import SceneConfig


def stats_transformer_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 2:
        # field added in version 3
        # since `scene_group` cannot be None, set it to a special value so that
        # `update_root()`, which is called by the predictor, knows to set
        # `stats_uri` to the old location of `stats.json`.
        cfg_dict['scene_group'] = '__N/A__'
    return cfg_dict


@register_config(
    'stats_transformer', upgrader=stats_transformer_config_upgrader)
class StatsTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.StatsTransformer`."""

    stats_uri: Optional[str] = Field(
        None,
        description='The URI of the output of the StatsAnalyzer. '
        'If None, and this Config is inside an RVPipeline, '
        'this field will be auto-generated.')
    scene_group: str = Field(
        'train_scenes',
        description='Name of the group of scenes whose stats to use. Defaults'
        'to "train_scenes".')

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None) -> None:
        if pipeline is not None and self.stats_uri is None:
            self.stats_uri = join(pipeline.analyze_uri, 'stats',
                                  self.scene_group, 'stats.json')

    def build(self):
        stats = RasterStats.load(self.stats_uri)
        return StatsTransformer(means=stats.means, stds=stats.stds)

    def update_root(self, root_dir: str) -> None:
        if self.scene_group == '__N/A__':
            # backward compatibility: use old location of stats.json
            self.stats_uri = join(root_dir, 'stats.json')
        else:
            self.stats_uri = join(root_dir, 'analyze', 'stats',
                                  self.scene_group, 'stats.json')
