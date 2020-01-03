import json

from rastervision.v2.data.vector_source.vector_source import VectorSource
from rastervision.v2.core.filesystem import file_to_str


class GeoJSONVectorSource(VectorSource):
    def __init__(self,
                 geojson_vs_config,
                 class_map,
                 crs_transformer
        super().__init__(geojson_vs_config, class_map, crs_transformer)

    def _get_geojson(self):
        geojson = json.loads(file_to_str(self.geojson_vs_config.uri))
        return self.class_inference.transform_geojson(geojson)
