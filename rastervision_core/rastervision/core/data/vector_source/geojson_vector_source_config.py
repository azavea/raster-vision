from typing import TYPE_CHECKING
from rastervision.core.data.vector_source import (VectorSourceConfig,
                                                  GeoJSONVectorSource)
from rastervision.pipeline.config import register_config, Field

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig, CRSTransformer)


def geojson_vector_source_config_upgrader(cfg_dict: dict,
                                          version: int) -> dict:
    if version == 7:
        cfg_dict['uris'] = cfg_dict.pop('uri', [])
    if version == 12:
        # removed in version 13
        cfg_dict.pop('ignore_crs_field', None)
    return cfg_dict


@register_config(
    'geojson_vector_source', upgrader=geojson_vector_source_config_upgrader)
class GeoJSONVectorSourceConfig(VectorSourceConfig):
    """Configure a :class:`.GeoJSONVectorSource`."""

    uris: str | list[str] = Field(
        ..., description='URI(s) of GeoJSON file(s).')

    def build(self,
              class_config: 'ClassConfig',
              crs_transformer: 'CRSTransformer',
              use_transformers: bool = True) -> GeoJSONVectorSource:
        if use_transformers:
            transformers = [
                tf.build(class_config=class_config) for tf in self.transformers
            ]
        else:
            transformers = []

        return GeoJSONVectorSource(
            uris=self.uris,
            crs_transformer=crs_transformer,
            vector_transformers=transformers)
