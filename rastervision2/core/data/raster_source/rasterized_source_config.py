from rastervision2.core.data.raster_source import (RasterizedSource)
from rastervision2.core.data.vector_source import (VectorSourceConfig)
from rastervision2.pipeline.config import register_config, Config


@register_config('rasterizer')
class RasterizerConfig(Config):
    background_class_id: int
    all_touched: bool = False


@register_config('rasterized_source')
class RasterizedSourceConfig(Config):
    vector_source: VectorSourceConfig
    rasterizer_config: RasterizerConfig

    def build(self, class_config, crs_transformer, extent):
        vector_source = self.vector_source.build(class_config, crs_transformer)

        return RasterizedSource(vector_source, self.rasterizer_config, extent,
                                crs_transformer)
