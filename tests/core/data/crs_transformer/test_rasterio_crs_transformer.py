import unittest

import numpy as np
import rasterio
from rasterio.windows import Window as RioWindow
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

    def test_map_to_pixel(self):
        # point tuple
        map_point = self.lon_lat
        pix_point = self.crs_trans.map_to_pixel(map_point)
        self.assertEqual(pix_point, self.pix_point)

        # array tuple
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

        # Box
        map_box = Box(*self.lon_lat[::-1], *self.lon_lat[::-1])
        pix_box = self.crs_trans.map_to_pixel(map_box)
        pix_box_expected = Box(*self.pix_point[::-1], *self.pix_point[::-1])
        self.assertEqual(pix_box, pix_box_expected)

        # shapely
        map_geom = Point(self.lon_lat)
        pix_geom = self.crs_trans.map_to_pixel(map_geom)
        pix_geom_expected = Point(self.pix_point)
        self.assertEqual(pix_geom, pix_geom_expected)

        # rasterio window
        map_box = Box(*self.lon_lat[::-1], *self.lon_lat[::-1])
        map_window = RioWindow.from_slices(
            *map_box.to_slices(), height=0, width=0)
        pix_window = self.crs_trans.map_to_pixel(map_window)
        pix_box_expected = Box(*self.pix_point[::-1], *self.pix_point[::-1])
        pix_window_expected = RioWindow.from_slices(
            *pix_box_expected.to_slices(), height=0, width=0)
        self.assertEqual(pix_window, pix_window_expected)

        # invalid input type
        self.assertRaises(TypeError,
                          lambda: self.crs_trans.map_to_pixel((1, 2, 3)))

    def test_pixel_to_map(self):
        # point tuple
        pix_point = self.pix_point
        map_point = self.crs_trans.pixel_to_map(pix_point)
        map_point_expected = self.lon_lat
        np.testing.assert_almost_equal(
            map_point, map_point_expected, decimal=3)

        # array tuple
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

        # Box
        pix_box = Box(*self.pix_point[::-1], *self.pix_point[::-1])
        map_box = self.crs_trans.pixel_to_map(pix_box)
        map_box_expected = Box(*self.lon_lat[::-1], *self.lon_lat[::-1])
        np.testing.assert_almost_equal(
            np.array(map_box.tuple_format()),
            np.array(map_box_expected.tuple_format()),
            decimal=3)

        # shapely
        pix_geom = Point(self.pix_point)
        map_geom = self.crs_trans.pixel_to_map(pix_geom)
        map_geom_expected = Point(self.lon_lat)
        np.testing.assert_almost_equal(
            np.concatenate(map_geom.xy).reshape(-1),
            np.concatenate(map_geom_expected.xy).reshape(-1),
            decimal=3)

        # rasterio window
        pix_box = Box(*self.pix_point[::-1], self.pix_point[1] + 10,
                      self.pix_point[0] + 10)
        pix_window = pix_box.to_rasterio()
        self.assertRaises(TypeError,
                          lambda: self.crs_trans.pixel_to_map(pix_window))

        # invalid input type
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
