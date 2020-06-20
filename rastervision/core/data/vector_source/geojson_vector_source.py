import json

from rastervision.core.data.vector_source.vector_source import VectorSource
from rastervision.pipeline.file_system import file_to_str


class GeoJSONVectorSource(VectorSource):
    def __init__(self, geojson_vs_config, class_config, crs_transformer):
        super().__init__(geojson_vs_config, class_config, crs_transformer)

    def _get_geojson(self):
        geojson = json.loads(file_to_str(self.vs_config.uri))
        if not self.vs_config.ignore_crs_field and 'crs' in geojson:
            raise Exception((
                'The GeoJSON file at {} contains a CRS field which is not '
                'allowed by the current GeoJSON standard or by Raster Vision. '
                'All coordinates are expected to be in EPSG:4326 CRS. If the file uses '
                'EPSG:4326 (ie. lat/lng on the WGS84 reference ellipsoid) and you would '
                'like to ignore the CRS field, set ignore_crs_field=True in '
                'GeoJSONVectorSourceConfig.').format(self.vs_config.uri))

        return self.class_inference.transform_geojson(geojson)
