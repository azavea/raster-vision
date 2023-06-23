import unittest
from os.path import join
from tempfile import NamedTemporaryFile

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core import (Box, RasterStats)
from rastervision.core.utils.misc import save_img
from rastervision.core.data.raster_source import (
    ChannelOrderError, RasterioSource, RasterioSourceConfig, fill_overflow)
from rastervision.core.data.raster_transformer import StatsTransformerConfig

from tests import data_file_path


class TestRasterioSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_nodata_val(self):
        # make geotiff filled with ones and zeros with nodata == 1
        img_path = join(self.tmp_dir, 'tmp.tif')
        height = 100
        width = 100
        nb_channels = 3
        with rasterio.open(
                img_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=nb_channels,
                dtype=np.uint8,
                nodata=1) as img_dataset:
            im = np.random.randint(0, 2, (height, width, nb_channels)).astype(
                np.uint8)
            for channel in range(nb_channels):
                img_dataset.write(im[:, :, channel], channel + 1)

        config = RasterioSourceConfig(uris=[img_path])
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_chip(source.extent)
        expected_out_chip = np.zeros((height, width, nb_channels))
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_mask(self):
        # make geotiff filled with ones and zeros and mask the whole image
        img_path = join(self.tmp_dir, 'tmp.tif')
        height = 100
        width = 100
        nb_channels = 3
        with rasterio.open(
                img_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=nb_channels,
                dtype=np.uint8) as img_dataset:
            im = np.random.randint(0, 2, (height, width, nb_channels)).astype(
                np.uint8)
            for channel in range(nb_channels):
                img_dataset.write(im[:, :, channel], channel + 1)
            img_dataset.write_mask(np.zeros(im.shape[0:2]).astype(bool))

        config = RasterioSourceConfig(uris=[img_path])
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_chip(source.extent)
        expected_out_chip = np.zeros((height, width, nb_channels))
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_dtype(self):
        img_path = data_file_path('small-rgb-tile.tif')
        config = RasterioSourceConfig(uris=[img_path])
        source = config.build(tmp_dir=self.tmp_dir)
        self.assertEqual(source.dtype, np.uint8)

    def test_gets_raw_chip(self):
        img_path = data_file_path('small-rgb-tile.tif')
        channel_order = [0, 1]

        config = RasterioSourceConfig(
            uris=[img_path], channel_order=channel_order)
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_raw_chip(source.extent)
        self.assertEqual(out_chip.shape[2], 3)

    def test_gets_raw_chip_from_uint16_transformed_proto(self):
        img_path = data_file_path('small-uint16-tile.tif')
        channel_order = [0, 1]

        config = RasterioSourceConfig(uris=[img_path])
        raw_rs = config.build(tmp_dir=self.tmp_dir)

        stats_uri = join(self.tmp_dir, 'tmp.tif')
        stats = RasterStats()
        stats.compute([raw_rs])
        stats.save(stats_uri)

        transformer = StatsTransformerConfig(stats_uri=stats_uri)
        config = RasterioSourceConfig(
            uris=[img_path],
            channel_order=channel_order,
            transformers=[transformer])
        rs = config.build(tmp_dir=self.tmp_dir)
        out_chip = rs.get_raw_chip(rs.extent)
        self.assertEqual(out_chip.shape[2], 3)

    def test_uses_channel_order(self):
        img_path = join(self.tmp_dir, 'img.tif')
        chip = np.ones((2, 2, 4)).astype(np.uint8)
        chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
        save_img(chip, img_path)

        channel_order = [0, 1, 2]
        config = RasterioSourceConfig(
            uris=[img_path], channel_order=channel_order)
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_chip(source.extent)
        expected_out_chip = np.ones((2, 2, 3)).astype(np.uint8)
        expected_out_chip[:, :, :] *= np.array([0, 1, 2]).astype(np.uint8)
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_channel_order_error(self):
        img_path = join(self.tmp_dir, 'img.tif')
        chip = np.ones((2, 2, 3)).astype(np.uint8)
        chip[:, :, :] *= np.array([0, 1, 2]).astype(np.uint8)
        save_img(chip, img_path)

        channel_order = [3, 1, 0]
        with self.assertRaises(ChannelOrderError):
            config = RasterioSourceConfig(
                uris=[img_path], channel_order=channel_order)
            config.build(tmp_dir=self.tmp_dir)

    def test_detects_alpha(self):
        # Set first channel to alpha. Expectation is that when omitting channel_order,
        # only the second and third channels will be in output.
        img_path = join(self.tmp_dir, 'img.tif')
        chip = np.ones((2, 2, 3)).astype(np.uint8)
        chip[:, :, :] *= np.array([0, 1, 2]).astype(np.uint8)
        save_img(chip, img_path)

        ci = (ColorInterp.alpha, ColorInterp.blue, ColorInterp.green)
        with rasterio.open(img_path, 'r+') as src:
            src.colorinterp = ci

        config = RasterioSourceConfig(uris=[img_path])
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_chip(source.extent)
        expected_out_chip = np.ones((2, 2, 2)).astype(np.uint8)
        expected_out_chip[:, :, :] *= np.array([1, 2]).astype(np.uint8)
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_non_geo(self):
        # Check if non-georeferenced image files can be read and CRSTransformer
        # implements the identity function.
        img_path = join(self.tmp_dir, 'img.png')
        chip = np.ones((2, 2, 3)).astype(np.uint8)
        save_img(chip, img_path)

        config = RasterioSourceConfig(uris=[img_path])
        source = config.build(tmp_dir=self.tmp_dir)
        out_chip = source.get_chip(source.extent)
        np.testing.assert_equal(out_chip, chip)

        p = (3, 4)
        out_p = source.crs_transformer.map_to_pixel(p)
        np.testing.assert_equal(out_p, p)

        out_p = source.crs_transformer.pixel_to_map(p)
        np.testing.assert_equal(out_p, p)

    def test_no_epsg(self):
        crs = rasterio.crs.CRS()
        img_path = join(self.tmp_dir, 'tmp.tif')
        height = 100
        width = 100
        nb_channels = 3
        with rasterio.open(
                img_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=nb_channels,
                dtype=np.uint8,
                crs=crs) as img_dataset:
            im = np.zeros((height, width, nb_channels)).astype(np.uint8)
            for channel in range(nb_channels):
                img_dataset.write(im[:, :, channel], channel + 1)

        try:
            config = RasterioSourceConfig(uris=[img_path])
            config.build(tmp_dir=self.tmp_dir)
        except Exception:
            self.fail(
                'Creating RasterioSource with CRS with no EPSG attribute '
                'raised an exception when it should not have.')

    def test_extent(self):
        img_path = data_file_path('small-rgb-tile.tif')
        cfg = RasterioSourceConfig(uris=[img_path])
        rs = cfg.build(tmp_dir=self.tmp_dir)
        self.assertEqual(rs.extent, Box(0, 0, 256, 256))

    def test_bbox(self):
        img_path = data_file_path('small-rgb-tile.tif')

        # /wo user specified bbox
        rs = RasterioSource(uris=img_path)
        self.assertEqual(rs.bbox, Box(0, 0, 256, 256))
        self.assertEqual(rs.extent, Box(0, 0, 256, 256))

        # /w user specified bbox
        rs_crop = RasterioSource(uris=img_path, bbox=Box(64, 64, 192, 192))

        # test bbox box
        self.assertEqual(rs_crop.bbox, Box(64, 64, 192, 192))
        self.assertEqual(rs_crop.extent, Box(0, 0, 128, 128))

        # test validators
        rs_cfg = RasterioSourceConfig(uris=[img_path], bbox=(0, 0, 1, 1))
        self.assertIsInstance(rs_cfg.bbox, Box)

    def test_fill_overflow(self):
        extent = Box(10, 10, 90, 90)
        window = Box(0, 0, 100, 100)
        arr = np.ones((100, 100, 1), dtype=np.uint8)
        out = fill_overflow(extent, window, arr)
        mask = np.zeros_like(arr).astype(bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(out[mask] == 1))
        self.assertTrue(np.all(out[~mask] == 0))

        window = Box(0, 0, 80, 100)
        arr = np.ones((80, 100, 1), dtype=np.uint8)
        out = fill_overflow(extent, window, arr)
        mask = np.zeros((80, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(out[mask] == 1))
        self.assertTrue(np.all(out[~mask] == 0))

    def test_extent_overflow(self):
        arr = np.ones((100, 100), dtype=np.uint8)
        with NamedTemporaryFile('wb') as fp:
            uri = fp.name
            with rasterio.open(
                    uri,
                    'w',
                    driver='GTiff',
                    height=100,
                    width=100,
                    count=1,
                    dtype=np.uint8) as ds:
                ds.write_band(1, arr)
            rs = RasterioSource(uris=uri, bbox=Box(10, 10, 90, 90))
            out = rs.get_chip(Box(0, 0, 100, 100))[..., 0]

        mask = np.zeros((100, 100), dtype=bool)
        mask[:80, :80] = 1
        self.assertTrue(np.all(out[mask] == 1))
        self.assertTrue(np.all(out[~mask] == 0))


if __name__ == '__main__':
    unittest.main()
