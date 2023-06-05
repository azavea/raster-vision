import unittest

import numpy as np
from shapely.geometry import (Polygon, MultiPolygon, Point, MultiPoint,
                              LineString, MultiLineString, mapping, shape)
import geopandas as gpd

from rastervision.core.box import Box
from rastervision.core.data.utils import (
    geometry_to_feature, geometries_to_geojson, is_empty_feature,
    remove_empty_features, split_multi_geometries, map_to_pixel_coords,
    pixel_to_map_coords, buffer_geoms, all_geoms_valid, geoms_to_geojson,
    merge_geojsons, geojson_to_geoms, geojson_to_geodataframe,
    get_geodataframe_extent, get_geojson_extent, filter_geojson_to_window,
    geoms_to_bbox_coords)
from tests.core.data.mock_crs_transformer import DoubleCRSTransformer


class TestGeojsonUtils(unittest.TestCase):
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

    def test_is_empty_feature(self):
        empty_feat = {'type': 'Feature'}
        non_empty_feat = geometry_to_feature(
            mapping(Polygon.from_bounds(0, 0, 10, 10)))
        self.assertTrue(is_empty_feature(empty_feat))
        self.assertFalse(is_empty_feature(non_empty_feat))

    def test_remove_empty_features(self):
        empty_feats = [{'type': 'Feature'}]
        non_empty_feats = [
            geometry_to_feature(mapping(Polygon.from_bounds(0, 0, 10, 10)))
        ]
        geojson = geometries_to_geojson(empty_feats + non_empty_feats)
        geojson_filtered = remove_empty_features(geojson)
        self.assertEqual(
            len(geojson_filtered['features']), len(non_empty_feats))

    def test_split_multi_geometries(self):
        # polygons
        geoms = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(20, 20, 30, 30)
        ]
        multi_feats = [geometry_to_feature(mapping(MultiPolygon(geoms)))]
        single_feats = [geometry_to_feature(mapping(g)) for g in geoms]
        geojson_in = geometries_to_geojson(multi_feats + single_feats)
        geojson_out = split_multi_geometries(geojson_in)
        self.assertEqual(len(geojson_out['features']), 4)

        # points
        geoms = [
            Point(0, 0),
            Point(1, 1),
        ]
        multi_feats = [geometry_to_feature(mapping(MultiPoint(geoms)))]
        single_feats = [geometry_to_feature(mapping(g)) for g in geoms]
        geojson_in = geometries_to_geojson(multi_feats + single_feats)
        geojson_out = split_multi_geometries(geojson_in)
        self.assertEqual(len(geojson_out['features']), 4)

        # linestrings
        geoms = [
            LineString([(0, 0), (1, 1), (2, 2)]),
            LineString([(0, 0), (2, 1), (4, 2)]),
        ]
        multi_feats = [geometry_to_feature(mapping(MultiLineString(geoms)))]
        single_feats = [geometry_to_feature(mapping(g)) for g in geoms]
        geojson_in = geometries_to_geojson(multi_feats + single_feats)
        geojson_out = split_multi_geometries(geojson_in)
        self.assertEqual(len(geojson_out['features']), 4)

    def test_map_to_pixel_coords(self):
        coords_in = np.array([1, 2])
        feat = mapping(Point(coords_in))
        crs_transformer = DoubleCRSTransformer()
        geojson_in = geometries_to_geojson([feat])
        geojson_out = map_to_pixel_coords(geojson_in, crs_transformer)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        coords_out = np.array(geom_out.coords).squeeze()
        np.testing.assert_array_almost_equal(coords_out, coords_in * 2)

    def test_pixel_to_map_coords(self):
        coords_in = np.array([1, 2])
        feat = mapping(Point(coords_in))
        crs_transformer = DoubleCRSTransformer()
        geojson_in = geometries_to_geojson([feat])
        geojson_out = pixel_to_map_coords(geojson_in, crs_transformer)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        coords_out = np.array(geom_out.coords).squeeze()
        np.testing.assert_array_almost_equal(coords_out, coords_in / 2)

    def test_simplify_polygons(self):
        """The buffer(0) trick in simplify_polygons() doesn't always work."""
        pass

    def test_all_geoms_valid(self):
        normal_polygon = Polygon.from_bounds(0, 0, 10, 10)
        bowtie_polygon = Polygon([(-1, 0), (1, 0), (1, 1), (-1, -1), (-1, 0)])
        # valid
        geojson_in = geometries_to_geojson([mapping(normal_polygon)])
        self.assertTrue(all_geoms_valid(geojson_in))
        # invalid
        geojson_in = geometries_to_geojson(
            [mapping(normal_polygon),
             mapping(bowtie_polygon)])
        self.assertFalse(all_geoms_valid(geojson_in))

    def test_buffer_geoms(self):
        class_bufs = {0: 5}
        properties = dict(class_id=0)

        # polygons
        geom_in = Polygon.from_bounds(0, 0, 10, 10)
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = buffer_geoms(
            geojson_in, geom_type='Polygon', class_bufs=class_bufs)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertEqual(geom_out, geom_in.buffer(5))

        # points
        geom_in = Point(0, 0)
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = buffer_geoms(
            geojson_in, geom_type='Point', class_bufs=class_bufs)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertEqual(geom_out, geom_in.buffer(5))

        # linestrings
        geom_in = LineString([(0, 0), (1, 1), (2, 2)])
        feat_in = geometry_to_feature(mapping(geom_in), properties)
        geojson_in = geometries_to_geojson([feat_in])
        geojson_out = buffer_geoms(
            geojson_in, geom_type='LineString', class_bufs=class_bufs)
        geom_out = shape(geojson_out['features'][0]['geometry'])
        self.assertEqual(geom_out, geom_in.buffer(5))

    def test_geoms_to_geojson_and_geojson_to_geoms(self):
        geoms_in = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(10, 10, 100, 100),
        ]
        geojson = geoms_to_geojson(geoms_in)
        geoms_out = list(geojson_to_geoms(geojson))
        self.assertListEqual(geoms_out, geoms_in)

    def test_merge_geojsons(self):
        geom_in_1 = Polygon.from_bounds(0, 0, 10, 10)
        geom_in_2 = Polygon.from_bounds(10, 10, 100, 100)
        geojson_1 = geoms_to_geojson([geom_in_1])
        geojson_2 = geoms_to_geojson([geom_in_2])
        geojson_merged = merge_geojsons([geojson_1, geojson_2])
        geom_out_1 = shape(geojson_merged['features'][0]['geometry'])
        geom_out_2 = shape(geojson_merged['features'][1]['geometry'])
        self.assertEqual(geom_out_1, geom_in_1)
        self.assertEqual(geom_out_2, geom_in_2)

    def test_geojson_to_geodataframe(self):
        geom = Polygon.from_bounds(0, 0, 10, 10)
        geojson = geoms_to_geojson([geom])
        gdf = geojson_to_geodataframe(geojson)
        self.assertIsInstance(gdf, gpd.GeoDataFrame)
        self.assertListEqual(list(gdf.geometry), [geom])

    def test_get_geodataframe_extent(self):
        geoms = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(20, 20, 30, 30),
        ]
        geojson = geoms_to_geojson(geoms)
        gdf = geojson_to_geodataframe(geojson)
        extent = get_geodataframe_extent(gdf)
        self.assertEqual(extent, Box(0, 0, 30, 30))

    def test_get_geojson_extent(self):
        geoms = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(20, 20, 30, 30),
        ]
        geojson = geoms_to_geojson(geoms)
        extent = get_geojson_extent(geojson)
        self.assertEqual(extent, Box(0, 0, 30, 30))

    def test_filter_geojson_to_window(self):
        geoms_in = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(20, 20, 30, 30),
        ]
        geojson_in = geoms_to_geojson(geoms_in)

        window = Box(0, 0, 15, 15)
        geojson_out = filter_geojson_to_window(geojson_in, window)
        geoms_out = list(geojson_to_geoms(geojson_out))
        self.assertListEqual(geoms_out, geoms_in[:1])

        window = Box(15, 15, 30, 30)
        geojson_out = filter_geojson_to_window(geojson_in, window)
        geoms_out = list(geojson_to_geoms(geojson_out))
        self.assertListEqual(geoms_out, geoms_in[1:])

        window = Box(0, 0, 5, 5)
        geojson_out = filter_geojson_to_window(geojson_in, window)
        geoms_out = list(geojson_to_geoms(geojson_out))
        self.assertListEqual(geoms_out, geoms_in[:1])

    def test_geoms_to_bbox_coords(self):
        geoms_in = [
            Polygon.from_bounds(0, 0, 10, 10),
            Polygon.from_bounds(20, 20, 30, 30),
        ]
        bbox = Box(10, 10, 20, 20)
        geoms_out = list(geoms_to_bbox_coords(geoms_in, bbox))
        geoms_out_expected = [
            Polygon.from_bounds(-10, -10, 0, 0),
            Polygon.from_bounds(10, 10, 20, 20),
        ]
        self.assertListEqual(geoms_out, geoms_out_expected)


if __name__ == '__main__':
    unittest.main()
