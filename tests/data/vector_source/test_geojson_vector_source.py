import unittest
import os

from shapely.geometry import shape

import rastervision as rv
from rastervision.data.vector_source import (GeoJSONVectorSourceConfigBuilder,
                                             GeoJSONVectorSourceConfig)
from rastervision.core.class_map import ClassMap
from rastervision.utils.files import json_to_file
from rastervision.rv_config import RVConfig
from rastervision.data.crs_transformer import IdentityCRSTransformer

from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestGeoJSONVectorSource(unittest.TestCase):
    """This also indirectly tests the ClassInference class."""

    def setUp(self):
        self.temp_dir = RVConfig.get_tmp_dir()
        self.uri = os.path.join(self.temp_dir.name, 'vectors.json')

    def tearDown(self):
        self.temp_dir.cleanup()

    def _test_class_inf(self, props, exp_class_ids, default_class_id=None):
        geojson = {
            'type':
            'FeatureCollection',
            'features': [{
                'properties': props,
                'geometry': {
                    'type': 'Point',
                    'coordinates': [1, 1]
                }
            }]
        }
        json_to_file(geojson, self.uri)

        class_map = ClassMap.construct_from(['building', 'car', 'tree'])
        class_id_to_filter = {
            1: ['==', 'type', 'building'],
            2: ['any', ['==', 'type', 'car'], ['==', 'type', 'auto']]
        }
        b = GeoJSONVectorSourceConfigBuilder() \
            .with_class_inference(class_id_to_filter=class_id_to_filter,
                                  default_class_id=default_class_id) \
            .with_uri(self.uri) \
            .build()
        msg = b.to_proto()
        config = GeoJSONVectorSourceConfig.from_proto(msg)
        source = config.create_source(
            crs_transformer=IdentityCRSTransformer(), class_map=class_map)
        trans_geojson = source.get_geojson()
        class_ids = [
            f['properties']['class_id'] for f in trans_geojson['features']
        ]
        self.assertEqual(class_ids, exp_class_ids)

    def test_class_inf_class_id(self):
        self._test_class_inf({'class_id': 3}, [3])

    def test_class_inf_label(self):
        self._test_class_inf({'label': 'car'}, [2])

    def test_class_inf_filter(self):
        self._test_class_inf({'type': 'auto'}, [2])

    def test_class_inf_default(self):
        self._test_class_inf({}, [4], default_class_id=4)

    def test_class_inf_no_default(self):
        self._test_class_inf({}, [])

    def geom_to_geojson(self, geom):
        return {'type': 'FeatureCollection', 'features': [{'geometry': geom}]}

    def transform_geojson(self,
                          geojson,
                          line_bufs=None,
                          point_bufs=None,
                          crs_transformer=None,
                          to_map_coords=False):
        if crs_transformer is None:
            crs_transformer = IdentityCRSTransformer()
        class_map = ClassMap.construct_from(['building'])
        json_to_file(geojson, self.uri)
        b = GeoJSONVectorSourceConfigBuilder() \
            .with_uri(self.uri) \
            .with_buffers(line_bufs=line_bufs, point_bufs=point_bufs) \
            .build()
        msg = b.to_proto()
        config = GeoJSONVectorSourceConfig.from_proto(msg)
        source = config.create_source(
            crs_transformer=crs_transformer, class_map=class_map)
        return source.get_geojson(to_map_coords=to_map_coords)

    def test_transform_geojson_no_coords(self):
        geom = {'type': 'Point', 'coordinates': []}
        geojson = self.geom_to_geojson(geom)
        trans_geojson = self.transform_geojson(geojson)

        self.assertEqual(0, len(trans_geojson['features']))

    def test_transform_geojson_geom_coll(self):
        geom = {
            'type':
            'GeometryCollection',
            'geometries': [{
                'type': 'MultiPoint',
                'coordinates': [[10, 10], [20, 20]]
            }]
        }
        geojson = self.geom_to_geojson(geom)
        trans_geojson = self.transform_geojson(geojson)

        feats = trans_geojson['features']
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0]['geometry']['type'], 'Polygon')
        self.assertEqual(feats[1]['geometry']['type'], 'Polygon')

    def test_transform_geojson_multi(self):
        geom = {'type': 'MultiPoint', 'coordinates': [[10, 10], [20, 20]]}
        geojson = self.geom_to_geojson(geom)
        trans_geojson = self.transform_geojson(geojson)

        feats = trans_geojson['features']
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0]['geometry']['type'], 'Polygon')
        self.assertEqual(feats[1]['geometry']['type'], 'Polygon')

    def test_transform_geojson_line_buf(self):
        geom = {'type': 'LineString', 'coordinates': [[10, 10], [10, 20]]}
        geojson = self.geom_to_geojson(geom)

        trans_geojson = self.transform_geojson(geojson, line_bufs={1: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(5.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, line_bufs={2: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(1.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, line_bufs={1: None})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

    def test_transform_point_buf(self):
        geom = {'type': 'Point', 'coordinates': [10, 10]}
        geojson = self.geom_to_geojson(geom)

        trans_geojson = self.transform_geojson(geojson, point_bufs={1: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(5.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, point_bufs={2: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(1.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, point_bufs={1: None})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

    def test_transform_polygon(self):
        geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]
        }
        geojson = self.geom_to_geojson(geom)

        trans_geojson = self.transform_geojson(geojson)
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(
            geojson, crs_transformer=DoubleCRSTransformer())
        trans_geom = trans_geojson['features'][0]['geometry']
        exp_geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [0, 20], [20, 20], [20, 0], [0, 0]]]
        }
        self.assertTrue(shape(exp_geom).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(
            geojson,
            crs_transformer=DoubleCRSTransformer(),
            to_map_coords=True)
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

    def test_validate_config(self):
        with self.assertRaises(rv.ConfigError):
            GeoJSONVectorSourceConfigBuilder() \
                .with_uri(self.uri) \
                .with_buffers(line_bufs={1: 'a'}) \
                .build()

        with self.assertRaises(rv.ConfigError):
            GeoJSONVectorSourceConfigBuilder() \
                .with_uri(self.uri) \
                .with_buffers(point_bufs={1: 'a'}) \
                .build()


if __name__ == '__main__':
    unittest.main()
