from typing import Optional
from os.path import join, basename

from rastervision2.pipeline.config import register_config
from rastervision2.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision2.core.data.raster_transformer.stats_transformer import (  # noqa
    StatsTransformer)
from rastervision2.core.raster_stats import RasterStats


@register_config('stats_transformer')
class StatsTransformerConfig(RasterTransformerConfig):
    stats_uri: Optional[str] = None

    def update(self, pipeline=None, scene=None):
        if pipeline is not None:
            self.stats_uri = join(pipeline.analyze_uri, 'stats.json')

    def build(self):
        return StatsTransformer(RasterStats.load(self.stats_uri))

    def update_root(self, root_dir):
        self.stats_uri = join(root_dir, basename(self.stats_uri))
