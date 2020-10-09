from typing import Dict

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.reclass_transformer import (  # noqa
    ReclassTransformer)


@register_config('reclass_transformer')
class ReclassTransformerConfig(RasterTransformerConfig):
    mapping: Dict[int, int] = Field(
        ..., description=('The reclassification mapping.'))

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and self.mapping is None:
            self.mapping = pipeline.mapping

    def build(self):
        return ReclassTransformer(mapping=self.mapping)
