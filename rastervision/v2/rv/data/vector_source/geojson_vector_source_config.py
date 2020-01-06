from rastervision.v2.rv.data.vector_source.vector_source_config import (
    VectorSourceConfig)
from rastervision.v2.rv.data.vector_source.geojson_vector_source import (
    GeoJSONVectorSource)
from rastervision.v2.core.config import register_config

@register_config('geojson_vector_source')
class GeoJSONVectorSourceConfig(VectorSourceConfig):
    uri: str

    def build(self, class_config, crs_transformer):
        return GeoJSONVectorSource(self, class_config, crs_transformer)
