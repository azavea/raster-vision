from rastervision.core.data.raster_source import (RasterizedSource)
from rastervision.core.data.vector_source import (VectorSourceConfig)
from rastervision.core.data.vector_transformer import (
    ClassInferenceTransformerConfig, BufferTransformerConfig)
from rastervision.pipeline.config import (register_config, Config, Field,
                                          validator)


@register_config('rasterizer')
class RasterizerConfig(Config):
    """Configure rasterization params for a :class:`.RasterizedSource`."""

    background_class_id: int = Field(
        ...,
        description='The class_id to use for any background pixels, i.e. '
        'pixels not covered by a polygon.')
    all_touched: bool = Field(
        False,
        description='If True, all pixels touched by geometries will be burned '
        'in. If false, only pixels whose center is within the polygon or that '
        'are selected by Bresenham\'s line algorithm will be burned in. '
        '(See rasterio.features.rasterize for more details).')


@register_config('rasterized_source')
class RasterizedSourceConfig(Config):
    """Configure a :class:`.RasterizedSource`."""

    vector_source: VectorSourceConfig
    rasterizer_config: RasterizerConfig

    @validator('vector_source')
    def ensure_required_transformers(
            cls, v: VectorSourceConfig) -> VectorSourceConfig:
        """Add class-inference and buffer transformers if absent."""
        tfs = v.transformers

        # add class inference transformer
        has_inf_tf = any(
            isinstance(tf, ClassInferenceTransformerConfig) for tf in tfs)
        if not has_inf_tf:
            tfs += [ClassInferenceTransformerConfig(default_class_id=None)]

        # add buffer transformers
        has_buf_tf = any(isinstance(tf, BufferTransformerConfig) for tf in tfs)
        if not has_buf_tf:
            tfs += [
                BufferTransformerConfig(geom_type='Point', default_buf=1),
                BufferTransformerConfig(geom_type='LineString', default_buf=1)
            ]

        return v

    def update(self, pipeline=None, scene=None):
        super().update(pipeline, scene)
        self.vector_source.update(pipeline, scene)

    def build(self, class_config, crs_transformer, extent):
        vector_source = self.vector_source.build(class_config, crs_transformer)
        return RasterizedSource(
            vector_source=vector_source,
            background_class_id=self.rasterizer_config.background_class_id,
            extent=extent,
            all_touched=self.rasterizer_config.all_touched)
