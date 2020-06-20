from rastervision.core.data.vector_source.vector_source_config import (
    VectorSourceConfig)
from rastervision.core.data.vector_source.geojson_vector_source import (
    GeoJSONVectorSource)
from rastervision.pipeline.config import register_config, Field


@register_config('geojson_vector_source')
class GeoJSONVectorSourceConfig(VectorSourceConfig):
    uri: str = Field(..., description='The URI of a GeoJSON file.')
    ignore_crs_field: bool = False

    def build(self, class_config, crs_transformer):
        return GeoJSONVectorSource(self, class_config, crs_transformer)
