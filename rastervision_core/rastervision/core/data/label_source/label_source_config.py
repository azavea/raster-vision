from typing import TYPE_CHECKING

from rastervision.pipeline.config import Config, register_config

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        LabelSource, SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig


@register_config('label_source')
class LabelSourceConfig(Config):
    """Configure a :class:`.LabelSource`."""

    def build(self,
              class_config: 'ClassConfig',
              crs_transformer: 'CRSTransformer',
              bbox: 'Box | None' = None,
              tmp_dir: str | None = None) -> 'LabelSource':
        raise NotImplementedError()

    def update(self,
               pipeline: 'RVPipelineConfig | None' = None,
               scene: 'SceneConfig | None' = None) -> None:
        pass
