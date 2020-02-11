import json

from rastervision2.core.data.vector_source.vector_source import VectorSource
from rastervision2.pipeline.filesystem import file_to_str


class GeoJSONVectorSource(VectorSource):
    def __init__(self, geojson_vs_config, class_config, crs_transformer):
        super().__init__(geojson_vs_config, class_config, crs_transformer)

    def _get_geojson(self):
        geojson = json.loads(file_to_str(self.vs_config.uri))
        return self.class_inference.transform_geojson(geojson)
