from rastervision2.core.data.raster_source import (RasterizedSource)
from rastervision2.core.data.vector_source import (VectorSourceConfig)
from rastervision2.pipeline.config import register_config, Config, Field


@register_config('rasterizer')
class RasterizerConfig(Config):
    background_class_id: int = Field(..., description=(
        'The class_id to use for any background pixels, ie. pixels not covered by a '
        'polygon.'))
    all_touched: bool = Field(False, description=(
        'If True, all pixels touched by geometries will be burned in. '
        'If false, only pixels whose center is within the polygon or '
        'that are selected by Bresenham’s line algorithm will be '
        'burned in. (See rasterio.features.rasterize).'))


@register_config('rasterized_source')
class RasterizedSourceConfig(Config):
    vector_source: VectorSourceConfig
    rasterizer_config: RasterizerConfig

    def build(self, class_config, crs_transformer, extent):
        vector_source = self.vector_source.build(class_config, crs_transformer)

        return RasterizedSource(vector_source, self.rasterizer_config, extent,
                                crs_transformer)
