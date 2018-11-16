import unittest
import json

from shapely.geometry import shape

from rastervision.data.vector_source import (MBTilesVectorSourceConfigBuilder,
                                             MBTilesVectorSourceConfig)
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap
from rastervision.data.crs_transformer import IdentityCRSTransformer
from rastervision.utils.files import file_to_str
from tests import data_file_path


class TestMBTilesVectorSource(unittest.TestCase):
    def setUp(self):
        self.uri = data_file_path('vector_tiles/{z}/{x}/{y}.mvt')
        self.class_id_to_filter = {1: ['has', 'building']}
        self.class_map = ClassMap.construct_from(['building'])
        self.crs_transformer = IdentityCRSTransformer()

        b = MBTilesVectorSourceConfigBuilder() \
            .with_class_inference(class_id_to_filter=self.class_id_to_filter,
                                  default_class_id=None) \
            .with_uri(self.uri) \
            .with_zoom(14) \
            .build()

        self.config = MBTilesVectorSourceConfig.from_proto(b.to_proto())

    def test_get_geojson(self):
        aoi_path = data_file_path('vector_tiles/lv-aoi.json')
        extent_geojson = json.loads(file_to_str(aoi_path))
        extent = Box.from_shapely(
            shape(extent_geojson['features'][0]['geometry']))
        source = self.config.create_source(self.crs_transformer, extent,
                                           self.class_map)
        geojson = source.get_geojson()
        expected_geojson = json.loads(
            file_to_str(data_file_path('vector_tiles/lv.json')))

        # Need to convert to JSON and back again because geojson object has tuples
        # instead of lists because of a quirk of shapely.geometry.mapping
        # See https://github.com/Toblerity/Shapely/issues/245
        geojson = json.loads(json.dumps(geojson))
        geojson['features'].sort(key=lambda f: f['properties']['__id'])
        expected_geojson['features'].sort(
            key=lambda f: f['properties']['__id'])

        self.assertDictEqual(geojson, expected_geojson)


if __name__ == '__main__':
    unittest.main()
