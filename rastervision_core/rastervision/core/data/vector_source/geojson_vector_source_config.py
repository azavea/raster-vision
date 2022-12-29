from typing import TYPE_CHECKING
from rastervision.core.data.vector_source import (VectorSourceConfig,
                                                  GeoJSONVectorSource)
from rastervision.pipeline.config import register_config, Field

if TYPE_CHECKING:
    from rastervision.core.data import (ClassConfig, CRSTransformer)


@register_config('geojson_vector_source')
class GeoJSONVectorSourceConfig(VectorSourceConfig):
    """Configure a :class:`.GeoJSONVectorSource`."""

    uri: str = Field(..., description='The URI of a GeoJSON file.')
    ignore_crs_field: bool = False

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
            uri=self.uri,
            ignore_crs_field=self.ignore_crs_field,
            crs_transformer=crs_transformer,
            vector_transformers=transformers)
