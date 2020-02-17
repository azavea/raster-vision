from typing import List

from rastervision2.pipeline.config import Config, register_config
from rastervision2.core.data.raster_transformer import RasterTransformerConfig


@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: List[int]
    transformers: List[RasterTransformerConfig] = []

    def build(self, tmp_dir, use_transformers=True):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
