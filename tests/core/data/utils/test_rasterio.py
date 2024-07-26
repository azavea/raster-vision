import unittest
from unittest.mock import patch
from os.path import join

import numpy as np
import pyproj

from rastervision.pipeline.file_system.utils import get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data.utils.rasterio import (
    crop_geotiff, get_aws_session, write_geotiff_like_geojson, write_bbox)
from rastervision.core.data import RasterioSource, GeoJSONVectorSource
from tests import data_file_path


class TestRasterioUtils(unittest.TestCase):
    def test_write_bbox(self):
        bbox = Box(ymin=48.815, xmin=2.224, ymax=48.902, xmax=2.469)
        crs_wkt = pyproj.CRS('epsg:4326').to_wkt()
        r = bbox.width / bbox.height
        arr1 = np.zeros((100, int(100 * r)))
        arr2 = np.zeros((100, int(100 * r), 4))
        with get_tmp_dir() as tmp_dir:
            geotiff_path = join(tmp_dir, 'test.geotiff')
            write_bbox(geotiff_path, arr1, bbox=bbox, crs_wkt=crs_wkt)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(
                rs.extent).normalize()
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)), np.array(list(bbox)), decimal=3)
            self.assertEqual(rs.shape, (*arr1.shape, 1))

            write_bbox(geotiff_path, arr2, bbox=bbox, crs_wkt=crs_wkt)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(
                rs.extent).normalize()
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)), np.array(list(bbox)), decimal=3)
            self.assertEqual(rs.shape, arr2.shape)

    def test_crop_geotiff(self):
        src_path = data_file_path('multi_raster_source/const_100_600x600.tiff')
        window = Box(0, 0, 10, 10)
        with get_tmp_dir() as tmp_dir:
            crop_path = join(tmp_dir, 'test.tiff')
            crop_geotiff(src_path, window, crop_path)
            rs = RasterioSource(crop_path)
            self.assertEqual(rs.extent, window)

    def test_write_geotiff_like_geojson(self):
        geojson_path = data_file_path('0-aoi.geojson')
        arr = np.zeros((10, 10))
        with get_tmp_dir() as tmp_dir:
            geotiff_path = join(tmp_dir, 'test.tiff')
            write_geotiff_like_geojson(
                geotiff_path, arr, geojson_path, crs=None)
            rs = RasterioSource(geotiff_path)
            geotiff_bbox = rs.crs_transformer.pixel_to_map(rs.extent)
            vs = GeoJSONVectorSource(geojson_path, rs.crs_transformer)
            geojson_bbox = rs.crs_transformer.pixel_to_map(vs.extent)
            np.testing.assert_array_almost_equal(
                np.array(list(geotiff_bbox)),
                np.array(list(geojson_bbox)),
                decimal=3)
            self.assertEqual(rs.shape, (10, 10, 1))

    @patch.dict('os.environ', AWS_REQUEST_PAYER='requester')
    def test_get_aws_session(self):
        session = get_aws_session()
        self.assertTrue(session.requester_pays)

    @patch.dict('sys.modules', {'rastervision.aws_s3': None})
    @patch.dict('os.environ', AWS_S3_REQUESTER_PAYS='false')
    def test_get_aws_session_no_rv_aws_s3_and_not_requester_pays(self):
        session = get_aws_session()
        self.assertFalse(session.requester_pays)


if __name__ == '__main__':
    unittest.main()
