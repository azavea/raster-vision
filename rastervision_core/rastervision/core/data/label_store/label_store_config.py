from typing import TYPE_CHECKING, Optional

from rastervision.pipeline.config import Config, register_config

if TYPE_CHECKING:
    from rastervision.core.box import Box
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        LabelStore, SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig


@register_config('label_store')
class LabelStoreConfig(Config):
    """Configure a :class:`.LabelStore`."""

    def build(self,
              class_config: 'ClassConfig',
              crs_transformer: 'CRSTransformer',
              bbox: Optional['Box'] = None,
              tmp_dir: Optional[str] = None) -> 'LabelStore':
        raise NotImplementedError()

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None) -> None:
        pass
