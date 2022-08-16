import unittest

import rasterio

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
        pix_point = self.crs_trans.map_to_pixel((self.lon_lat))
        self.assertEqual(pix_point, self.pix_point)

    def test_pixel_to_map(self):
        lon_lat = self.crs_trans.pixel_to_map((self.pix_point))
        self.assertAlmostEqual(lon_lat[0], self.lon_lat[0], places=3)
        self.assertAlmostEqual(lon_lat[1], self.lon_lat[1], places=3)

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
