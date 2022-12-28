from typing import Dict

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       ReclassTransformer)


@register_config('reclass_transformer')
class ReclassTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.ReclassTransformer`."""

    mapping: Dict[int, int] = Field(
        ..., description=('The reclassification mapping.'))

    def build(self) -> ReclassTransformer:
        return ReclassTransformer(mapping=self.mapping)
