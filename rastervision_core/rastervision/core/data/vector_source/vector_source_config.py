from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

from rastervision.pipeline.config import Config, register_config, Field
from rastervision.core.data.vector_transformer import VectorTransformerConfig

if TYPE_CHECKING:
    from rastervision.core.rv_pipeline import RVPipelineConfig
    from rastervision.core.data import (ClassConfig, CRSTransformer,
                                        SceneConfig, VectorSource)


def vector_source_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 4:
        from rastervision.core.data.vector_transformer import (
            ClassInferenceTransformerConfig, BufferTransformerConfig)

        class_inf_tf = ClassInferenceTransformerConfig(
            default_class_id=cfg_dict.get('default_class_id'),
            class_id_to_filter=cfg_dict.get('class_id_to_filter')).dict()

        line_bufs = {} if cfg_dict.get('line_bufs') is None else cfg_dict.get(
            'line_bufs')
        point_bufs = {} if cfg_dict.get(
            'point_bufs') is None else cfg_dict.get('point_bufs')
        buf_tfs = [
            BufferTransformerConfig(
                geom_type='LineString', class_bufs=line_bufs).dict(),
            BufferTransformerConfig(geom_type='Point',
                                    class_bufs=point_bufs).dict()
        ]
        # added in version 5
        cfg_dict['transformers'] = [class_inf_tf, *buf_tfs]
        try:
            # removed in version 5
            del cfg_dict['default_class_id']
            del cfg_dict['class_id_to_filter']
            del cfg_dict['line_bufs']
            del cfg_dict['point_bufs']
        except KeyError:
            pass
    return cfg_dict


@register_config('vector_source', upgrader=vector_source_config_upgrader)
class VectorSourceConfig(Config):
    """Configure a :class:`.VectorSource`."""

    transformers: List[VectorTransformerConfig] = Field(
        [], description='List of VectorTransformers.')

    @abstractmethod
    def build(self, class_config: 'ClassConfig',
              crs_transformer: 'CRSTransformer') -> 'VectorSource':
        pass

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None) -> None:
        pass
