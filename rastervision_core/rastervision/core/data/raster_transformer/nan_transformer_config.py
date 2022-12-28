from typing import Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.nan_transformer import (  # noqa
    NanTransformer)


@register_config('nan_transformer')
class NanTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.NanTransformer`."""

    to_value: Optional[float] = Field(
        0.0, description=('Turn all NaN values into this value.'))

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and self.to_value is None:
            self.to_value = pipeline.to_value

    def build(self):
        return NanTransformer(to_value=self.to_value)
