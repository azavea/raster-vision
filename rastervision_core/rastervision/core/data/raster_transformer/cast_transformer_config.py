from typing import Optional

from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.raster_transformer.raster_transformer_config import (  # noqa
    RasterTransformerConfig)
from rastervision.core.data.raster_transformer.cast_transformer import (  # noqa
    CastTransformer)


@register_config('cast_transformer')
class CastTransformerConfig(RasterTransformerConfig):
    to_dtype: Optional[str] = Field(
        'np.uint8', description=('dtype to cast raster to.'))

    def update(self, pipeline=None, scene=None):
        if pipeline is not None and self.to_dtype is None:
            self.to_dtype = pipeline.to_dtype

    def build(self):
        return CastTransformer(to_dtype=self.to_dtype)
