from typing import List, Optional

from rastervision.pipeline.config import Config, register_config, Field
from rastervision.core.data.raster_transformer import RasterTransformerConfig


@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: Optional[List[int]] = Field(
        None,
        description=
        'The sequence of channel indices to use when reading imagery.')
    transformers: List[RasterTransformerConfig] = []

    def build(self, tmp_dir, use_transformers=True):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
