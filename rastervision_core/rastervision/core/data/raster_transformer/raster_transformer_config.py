from rastervision.pipeline.config import Config, register_config


@register_config('raster_transformer')
class RasterTransformerConfig(Config):
    """Configure a :class:`.RasterTransformer`."""

    def update(self, pipeline=None, scene=None):
        pass

    def update_root(self, root_dir):
        pass
