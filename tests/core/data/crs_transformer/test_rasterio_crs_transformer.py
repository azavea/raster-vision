import unittest

import numpy as np
import rasterio
from shapely.geometry import Point

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from tests import data_file_path


class TestRasterioCRSTransformer(unittest.TestCase):
    def setUp(self):
        self.im_path = data_file_path('3857.tif')
        self.im_dataset = rasterio.open(self.im_path)
        self.crs_trans = RasterioCRSTransformer.from_dataset(self.im_dataset)
        self.lon_lat = (-115.3063715, 36.1268253)
        self.pix_point = (50, 61)

    def test_map_to_pixel_point(self):
        # w/o bbox
        map_point = self.lon_lat
        pix_point = self.crs_trans.map_to_pixel(map_point)
        self.assertEqual(pix_point, self.pix_point)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        map_point = self.lon_lat
        pix_point = self.crs_trans.map_to_pixel(map_point, bbox=bbox)
        pix_x, pix_y = self.pix_point
        pix_point_expected = (pix_x - 20, pix_y - 20)
        self.assertEqual(pix_point, pix_point_expected)

    def test_map_to_pixel_array(self):
        # w/o bbox
        map_point = np.array([
            self.lon_lat,
            self.lon_lat,
        ])
        pix_point = self.crs_trans.map_to_pixel((map_point[:, 0],
                                                 map_point[:, 1]))
        pix_point_expected = np.array([
            self.pix_point,
            self.pix_point,
        ])
        np.testing.assert_equal(pix_point[0], pix_point_expected[:, 0])
        np.testing.assert_equal(pix_point[1], pix_point_expected[:, 1])

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        map_point = np.array([
            self.lon_lat,
            self.lon_lat,
        ])
        pix_point = self.crs_trans.map_to_pixel(
            (map_point[:, 0], map_point[:, 1]), bbox=bbox)
        pix_point_expected = np.array([
            self.pix_point,
            self.pix_point,
        ])
        pix_point_expected -= 20
        np.testing.assert_equal(pix_point[0], pix_point_expected[:, 0])
        np.testing.assert_equal(pix_point[1], pix_point_expected[:, 1])

    def test_map_to_pixel_box(self):
        # w/o bbox
        map_x, map_y = self.lon_lat
        map_box = Box(map_y, map_x, map_y, map_x)
        pix_box = self.crs_trans.map_to_pixel(map_box)
        pix_x, pix_y = self.pix_point
        pix_box_expected = Box(pix_y, pix_x, pix_y, pix_x)
        self.assertEqual(pix_box, pix_box_expected)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        map_x, map_y = self.lon_lat
        map_box = Box(map_y, map_x, map_y, map_x)
        pix_box = self.crs_trans.map_to_pixel(map_box, bbox=bbox)
        pix_x, pix_y = self.pix_point
        pix_box_expected = Box(pix_y - 20, pix_x - 20, pix_y - 20, pix_x - 20)
        self.assertEqual(pix_box, pix_box_expected)

    def test_map_to_pixel_shapely(self):
        # w/o bbox
        map_geom = Point(self.lon_lat)
        pix_geom = self.crs_trans.map_to_pixel(map_geom)
        pix_geom_expected = Point(self.pix_point)
        self.assertEqual(pix_geom, pix_geom_expected)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        map_geom = Point(self.lon_lat)
        pix_geom = self.crs_trans.map_to_pixel(map_geom, bbox=bbox)
        pix_x, pix_y = self.pix_point
        pix_geom_expected = Point((pix_x - 20, pix_y - 20))
        self.assertEqual(pix_geom, pix_geom_expected)

    def test_map_to_pixel_invalid_input(self):
        self.assertRaises(TypeError,
                          lambda: self.crs_trans.map_to_pixel((1, 2, 3)))

    def test_pixel_to_map_point(self):
        # w/o bbox
        pix_point = self.pix_point
        map_point = self.crs_trans.pixel_to_map(pix_point)
        map_point_expected = self.lon_lat
        np.testing.assert_almost_equal(
            map_point, map_point_expected, decimal=3)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        pix_x, pix_y = self.pix_point
        pix_point = (pix_x - 20, pix_y - 20)
        map_point = self.crs_trans.pixel_to_map(pix_point, bbox=bbox)
        map_point_expected = self.lon_lat
        np.testing.assert_almost_equal(
            map_point, map_point_expected, decimal=3)

    def test_pixel_to_map_array(self):
        # w/o bbox
        pix_point = np.array([
            self.pix_point,
            self.pix_point,
        ])
        map_point = self.crs_trans.pixel_to_map((pix_point[:, 0],
                                                 pix_point[:, 1]))
        map_point_expected = np.array([
            self.lon_lat,
            self.lon_lat,
        ])
        np.testing.assert_almost_equal(
            map_point[0], map_point_expected[:, 0], decimal=3)
        np.testing.assert_almost_equal(
            map_point[1], map_point_expected[:, 1], decimal=3)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        pix_point = np.array([
            self.pix_point,
            self.pix_point,
        ])
        pix_point -= 20
        map_point = self.crs_trans.pixel_to_map(
            (pix_point[:, 0], pix_point[:, 1]), bbox=bbox)
        map_point_expected = np.array([
            self.lon_lat,
            self.lon_lat,
        ])
        np.testing.assert_almost_equal(
            map_point[0], map_point_expected[:, 0], decimal=3)
        np.testing.assert_almost_equal(
            map_point[1], map_point_expected[:, 1], decimal=3)

    def test_pixel_to_map_box(self):
        # w/o bbox
        pix_x, pix_y = self.pix_point
        pix_box = Box(pix_y, pix_x, pix_y, pix_x)
        map_box = self.crs_trans.pixel_to_map(pix_box)
        map_x, map_y = self.lon_lat
        map_box_expected = Box(map_y, map_x, map_y, map_x)
        np.testing.assert_almost_equal(
            np.array(map_box.tuple_format()),
            np.array(map_box_expected.tuple_format()),
            decimal=3)

        # w/ bbox
        bbox = Box(20, 20, 80, 80)
        pix_x, pix_y = self.pix_point
        pix_box = Box(pix_y - 20, pix_x - 20, pix_y - 20, pix_x - 20)
        map_box = self.crs_trans.pixel_to_map(pix_box, bbox=bbox)
        map_x, map_y = self.lon_lat
        map_box_expected = Box(map_y, map_x, map_y, map_x)
        np.testing.assert_almost_equal(
            np.array(map_box.tuple_format()),
            np.array(map_box_expected.tuple_format()),
            decimal=3)

    def test_pixel_to_map_shapely(self):
        # w/o bbox
        pix_geom = Point(self.pix_point)
        map_geom = self.crs_trans.pixel_to_map(pix_geom)
        map_geom_expected = Point(self.lon_lat)
        np.testing.assert_almost_equal(
            np.concatenate(map_geom.xy).reshape(-1),
            np.concatenate(map_geom_expected.xy).reshape(-1),
            decimal=3)

        # w/o bbox
        bbox = Box(20, 20, 80, 80)
        pix_x, pix_y = self.pix_point
        pix_geom = Point((pix_x - 20, pix_y - 20))
        map_geom = self.crs_trans.pixel_to_map(pix_geom, bbox=bbox)
        map_geom_expected = Point(self.lon_lat)
        np.testing.assert_almost_equal(
            np.concatenate(map_geom.xy).reshape(-1),
            np.concatenate(map_geom_expected.xy).reshape(-1),
            decimal=3)

    def test_pixel_to_map_invalid_input(self):
        self.assertRaises(TypeError,
                          lambda: self.crs_trans.pixel_to_map((1, 2, 3)))

    def test_from_dataset(self):
        # default map_crs
        tf = RasterioCRSTransformer.from_dataset(self.im_dataset)
        im_crs = self.im_dataset.crs.wkt.lower()
        self.assertEqual(tf.map_crs.lower(), 'epsg:4326')
        self.assertEqual(tf.image_crs.lower(), self.im_dataset.crs.wkt.lower())

        # map_crs = None
        tf = RasterioCRSTransformer.from_dataset(self.im_dataset, map_crs=None)
        im_crs = self.im_dataset.crs.wkt.lower()
        self.assertEqual(tf.map_crs.lower(), im_crs)
        self.assertEqual(tf.image_crs.lower(), im_crs)

    def test_from_uri(self):
        # default map_crs
        tf = RasterioCRSTransformer.from_uri(self.im_path)
        im_crs = self.im_dataset.crs.wkt.lower()
        self.assertEqual(tf.map_crs.lower(), 'epsg:4326')
        self.assertEqual(tf.image_crs.lower(), self.im_dataset.crs.wkt.lower())

        # map_crs = None
        tf = RasterioCRSTransformer.from_uri(self.im_path, map_crs=None)
        im_crs = self.im_dataset.crs.wkt.lower()
        self.assertEqual(tf.map_crs.lower(), im_crs)
        self.assertEqual(tf.image_crs.lower(), im_crs)


if __name__ == '__main__':
    unittest.main()
