from rastervision.pipeline.config import register_config, Field
from rastervision.core.data.class_config import ClassConfig
from rastervision.core.data.raster_transformer import (RasterTransformerConfig,
                                                       RGBClassTransformer)


@register_config('rgb_class_transformer')
class RGBClassTransformerConfig(RasterTransformerConfig):
    """Configure a :class:`.RGBClassTransformer`."""

    class_config: ClassConfig = Field(
        ...,
        description=('The class config defining the mapping between '
                     'classes and colors.'))

    def build(self) -> RGBClassTransformer:
        return RGBClassTransformer(class_config=self.class_config)
