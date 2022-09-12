from rastervision.pipeline.config import register_config
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       MinMaxTransformer)


@register_config('min_max_transformer')
class MinMaxTransformerConfig(RasterTransformerConfig):
    """Transforms chips by scaling values in each channel to span 0-255."""

    def build(self):
        return MinMaxTransformer()
