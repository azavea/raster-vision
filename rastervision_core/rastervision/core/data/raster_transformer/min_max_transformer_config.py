from rastervision.pipeline.config import register_config
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       MinMaxTransformer)


@register_config('min_max_transformer')
class MinMaxTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.MinMaxTransformer`."""

    def build(self):
        return MinMaxTransformer()
