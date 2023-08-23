from typing import TYPE_CHECKING
from rastervision.pipeline.config import Config, register_config

if TYPE_CHECKING:
    from rastervision.core.data import SceneConfig
    from rastervision.core.rv_pipeline import RVPipelineConfig


@register_config('raster_transformer')
class RasterTransformerConfig(Config):
    """Configure a :class:`.RasterTransformer`."""

    def update(self,
               pipeline: 'RVPipelineConfig' = None,
               scene: 'SceneConfig' = None):
        pass

    def update_root(self, root_dir: str):
        pass
