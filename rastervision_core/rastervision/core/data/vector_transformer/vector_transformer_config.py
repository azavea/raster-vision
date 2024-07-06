from abc import abstractmethod
from typing import TYPE_CHECKING

from rastervision.pipeline.config import Config, register_config

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.core.data import ClassConfig, SceneConfig
    from rastervision.core.data.vector_transformer import VectorTransformer


@register_config('vector_transformer')
class VectorTransformerConfig(Config):
    """Configure a :class:`.VectorTransformer`."""

    def update(self,
               pipeline: 'RVPipelineConfig | None' = None,
               scene: 'SceneConfig | None' = None) -> None:
        pass

    @abstractmethod
    def build(self, class_config: 'ClassConfig') -> 'VectorTransformer':
        pass
