import unittest

from rastervision.core.data.utils import (geometry_to_feature,
                                          geometries_to_geojson)


class TestUtils(unittest.TestCase):
    def test_geometry_to_feature(self):
        geometry = {'type': 'Polygon', 'coordinates': []}
        feature = geometry_to_feature(geometry)
        self.assertTrue(feature['type'], 'Feature')
        self.assertTrue('geometry' in feature)
        self.assertEqual(feature['geometry'], geometry)

        feature2 = geometry_to_feature(feature)
        self.assertEqual(feature2, feature)

    def test_geometries_to_geojson(self):
        geometries = [{'type': 'Polygon', 'coordinates': []}]
        geojson = geometries_to_geojson(geometries)
        self.assertTrue(geojson['type'], 'FeatureCollection')
        self.assertTrue('features' in geojson)
        self.assertTrue('geometry' in geojson['features'][0])
        self.assertEqual(geojson['features'][0]['geometry'], geometries[0])


if __name__ == '__main__':
    unittest.main()
