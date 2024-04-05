from typing import Callable
import unittest
from os.path import join

from shapely.geometry import shape
import geopandas as gpd

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import (
    BufferTransformerConfig, ClassConfig, ClassInferenceTransformerConfig,
    GeoJSONVectorSource, GeoJSONVectorSourceConfig, IdentityCRSTransformer)
from rastervision.core.data.vector_source.geojson_vector_source_config import (
    geojson_vector_source_config_upgrader)
from rastervision.core.data.utils import geometries_to_geojson

from tests import test_config_upgrader
from tests.core.data.mock_crs_transformer import DoubleCRSTransformer


class TestGeoJSONVectorSourceConfig(unittest.TestCase):
    def test_upgrader_v8(self):
        cfg = GeoJSONVectorSourceConfig(uris=['a', 'b'])
        old_cfg_dict = cfg.dict()
        old_cfg_dict['uri'] = old_cfg_dict['uris']
        del old_cfg_dict['uris']
        test_config_upgrader(
            cfg_class=GeoJSONVectorSourceConfig,
            old_cfg_dict=old_cfg_dict,
            upgrader=geojson_vector_source_config_upgrader,
            curr_version=8)

    def test_upgrader_v13(self):
        cfg = GeoJSONVectorSourceConfig(uris=['a', 'b'])
        old_cfg_dict = cfg.dict()
        old_cfg_dict['ignore_crs_field'] = True
        test_config_upgrader(
            cfg_class=GeoJSONVectorSourceConfig,
            old_cfg_dict=old_cfg_dict,
            upgrader=geojson_vector_source_config_upgrader,
            curr_version=13)


class TestGeoJSONVectorSource(unittest.TestCase):
    """This also indirectly tests the ClassInference class."""

    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def setUp(self):
        self.tmp_dir = get_tmp_dir()
        self.uri = join(self.tmp_dir.name, 'vectors.json')

    def tearDown(self):
        self.tmp_dir.cleanup()

    def transform_geojson(self,
                          geojson,
                          line_bufs={},
                          point_bufs={},
                          crs_transformer=None,
                          to_map_coords=False):
        if crs_transformer is None:
            crs_transformer = IdentityCRSTransformer()
        class_config = ClassConfig(names=['building'])
        json_to_file(geojson, self.uri)
        cfg = GeoJSONVectorSourceConfig(
            uris=self.uri,
            transformers=[
                ClassInferenceTransformerConfig(default_class_id=0),
                BufferTransformerConfig(
                    geom_type='LineString', class_bufs=line_bufs),
                BufferTransformerConfig(
                    geom_type='Point', class_bufs=point_bufs)
            ])
        source = cfg.build(class_config, crs_transformer)
        return source.get_geojson(to_map_coords=to_map_coords)

    def test_transform_geojson_no_coords(self):
        geom = {'type': 'Point', 'coordinates': []}
        geojson = geometries_to_geojson([geom])
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
        geojson = geometries_to_geojson([geom])
        trans_geojson = self.transform_geojson(geojson)

        feats = trans_geojson['features']
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0]['geometry']['type'], 'Polygon')
        self.assertEqual(feats[1]['geometry']['type'], 'Polygon')

    def test_transform_geojson_multi(self):
        geom = {'type': 'MultiPoint', 'coordinates': [[10, 10], [20, 20]]}
        geojson = geometries_to_geojson([geom])
        trans_geojson = self.transform_geojson(geojson)

        feats = trans_geojson['features']
        self.assertEqual(len(feats), 2)
        self.assertEqual(feats[0]['geometry']['type'], 'Polygon')
        self.assertEqual(feats[1]['geometry']['type'], 'Polygon')

    def test_transform_geojson_line_buf(self):
        geom = {'type': 'LineString', 'coordinates': [[10, 10], [10, 20]]}
        geojson = geometries_to_geojson([geom])

        trans_geojson = self.transform_geojson(geojson, line_bufs={0: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(5.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, line_bufs={1: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(1.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, line_bufs={0: None})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

    def test_transform_point_buf(self):
        geom = {'type': 'Point', 'coordinates': [10, 10]}
        geojson = geometries_to_geojson([geom])

        trans_geojson = self.transform_geojson(geojson, point_bufs={0: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(5.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, point_bufs={1: 5.0})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).buffer(1.0).equals(shape(trans_geom)))

        trans_geojson = self.transform_geojson(geojson, point_bufs={0: None})
        trans_geom = trans_geojson['features'][0]['geometry']
        self.assertTrue(shape(geom).equals(shape(trans_geom)))

    def test_transform_polygon(self):
        geom = {
            'type': 'Polygon',
            'coordinates': [[[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]]
        }
        geojson = geometries_to_geojson([geom])

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

    def test_crs_in_geojson(self):
        geom_4326 = Box(10., 10., 20., 20.).to_shapely()
        gdf_4326 = gpd.GeoDataFrame(geometry=[geom_4326], crs='epsg:4326')
        gdf_3857 = gdf_4326.to_crs('epsg:3857')
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, 'test_3857.json')
            gdf_3857.to_file(path)
            vs = GeoJSONVectorSource(
                path, crs_transformer=IdentityCRSTransformer())
            geom_4326_out = vs.get_geoms()[0]
        self.assertEqual(geom_4326_out, geom_4326)

    def test__geo_interface__(self):
        geom_4326 = Box(10., 10., 20., 20.).to_shapely()
        gdf_4326 = gpd.GeoDataFrame(geometry=[geom_4326], crs='epsg:4326')
        gdf_3857 = gdf_4326.to_crs('epsg:3857')
        with get_tmp_dir() as tmp_dir:
            path = join(tmp_dir, 'test_3857.json')
            gdf_3857.to_file(path)
            vs = GeoJSONVectorSource(
                path, crs_transformer=IdentityCRSTransformer())
            self.assertNoError(lambda: gpd.GeoDataFrame.from_features(vs))


if __name__ == '__main__':
    unittest.main()
