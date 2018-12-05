import unittest
import json

from shapely.geometry import shape

from rastervision.data.vector_source import (
    VectorTileVectorSourceConfigBuilder, VectorTileVectorSourceConfig)
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.data.crs_transformer import IdentityCRSTransformer
from rastervision.utils.files import file_to_str
from tests import data_file_path


class TestVectorTileVectorSource(unittest.TestCase):
    def setUp(self):
        self.class_id_to_filter = {1: ['has', 'building']}
        self.class_map = ClassMap.construct_from(['building'])
        self.crs_transformer = IdentityCRSTransformer()

    def _get_source(self, uri):
        b = VectorTileVectorSourceConfigBuilder() \
            .with_class_inference(class_id_to_filter=self.class_id_to_filter,
                                  default_class_id=None) \
            .with_uri(uri) \
            .with_zoom(14) \
            .with_id_field('__id') \
            .build()
        config = VectorTileVectorSourceConfig.from_proto(b.to_proto())
        aoi_path = data_file_path('vector_tiles/lv-aoi.json')
        extent_geojson = json.loads(file_to_str(aoi_path))
        extent = Box.from_shapely(
            shape(extent_geojson['features'][0]['geometry']))
        source = config.create_source(self.crs_transformer, extent,
                                      self.class_map)
        return source

    def _test_get_geojson(self, vector_tile_uri, json_uri):
        source = self._get_source(vector_tile_uri)
        geojson = source.get_geojson()
        expected_geojson = json.loads(file_to_str(data_file_path(json_uri)))

        # Need to convert to JSON and back again because geojson object has tuples
        # instead of lists because of a quirk of shapely.geometry.mapping
        # See https://github.com/Toblerity/Shapely/issues/245
        geojson = json.loads(json.dumps(geojson))
        geojson['features'].sort(key=lambda f: f['properties']['__id'])
        expected_geojson['features'].sort(
            key=lambda f: f['properties']['__id'])

        self.assertDictEqual(geojson, expected_geojson)

    def test_get_geojson_from_zxy(self):
        vector_tile_uri = data_file_path('vector_tiles/{z}/{x}/{y}.mvt')
        json_uri = 'vector_tiles/lv-zxy.json'
        self._test_get_geojson(vector_tile_uri, json_uri)

    def test_get_geojson_from_mbtiles(self):
        vector_tile_uri = data_file_path('vector_tiles/lv.mbtiles')
        json_uri = 'vector_tiles/lv-mbtiles.json'
        self._test_get_geojson(vector_tile_uri, json_uri)


if __name__ == '__main__':
    unittest.main()
