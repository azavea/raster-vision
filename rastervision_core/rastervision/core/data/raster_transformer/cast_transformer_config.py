from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.cast_transformer import (  # noqa
    CastTransformer)


@register_config('cast_transformer')
class CastTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.CastTransformer`."""

    to_dtype: str = Field(
        ...,
        description='dtype to cast raster to. Must be a valid Numpy dtype '
        'e.g. "uint8", "float32", etc.')

    def build(self):
        return CastTransformer(to_dtype=self.to_dtype)
