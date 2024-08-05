from typing import TYPE_CHECKING
from os.path import join

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       StatsTransformer)

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
    elif version == 13:
        cfg_dict['needs_channel_order'] = True
    return cfg_dict


@register_config(
    'stats_transformer', upgrader=stats_transformer_config_upgrader)
class StatsTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.StatsTransformer`."""

    stats_uri: str | None = Field(
        None,
        description='The URI of the output of the StatsAnalyzer. '
        'If None, and this Config is inside an RVPipeline, '
        'this field will be auto-generated.')
    scene_group: str = Field(
        'train_scenes',
        description='Name of the group of scenes whose stats to use. Defaults'
        'to "train_scenes".')
    needs_channel_order: bool = Field(
        False,
        description='Whether the means and stds in the stats_uri file need to '
        'be re-ordered/subsetted using ``channel_order`` to be compatible '
        'with the chips that will be passed to the :class:`.StatsTransformer` '
        'by the :class:`.RasterSource`. This field exists for backward '
        'compatibility with Raster Vision versions <= 0.30. It will be set '
        'automatically when loading stats from older model-bundles.')

    def update(self,
               pipeline: 'RVPipelineConfig | None' = None,
               scene: 'SceneConfig | None' = None) -> None:
        if pipeline is not None and self.stats_uri is None:
            self.stats_uri = join(pipeline.analyze_uri, 'stats',
                                  self.scene_group, 'stats.json')

    def build(self,
              channel_order: list[int] | None = None) -> StatsTransformer:
        if self.needs_channel_order:
            tf = StatsTransformer.from_stats_json(
                self.stats_uri, channel_order=channel_order)
        else:
            tf = StatsTransformer.from_stats_json(self.stats_uri)
        return tf

    def update_root(self, root_dir: str) -> None:
        if self.scene_group == '__N/A__':
            # backward compatibility: use old location of stats.json
            self.stats_uri = join(root_dir, 'stats.json')
        else:
            self.stats_uri = join(root_dir, 'analyze', 'stats',
                                  self.scene_group, 'stats.json')
