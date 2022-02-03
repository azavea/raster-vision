import unittest
from pydantic import ValidationError

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import (RasterioSourceConfig,
                                    MultiRasterSourceConfig,
                                    SubRasterSourceConfig, CropOffsets)
from rastervision.pipeline import rv_config

from tests import data_file_path


def make_cfg(img_path='small-rgb-tile.tif', **kwargs):
    img_path = data_file_path(img_path)
    r_source = RasterioSourceConfig(uris=[img_path], channel_order=[0])
    g_source = RasterioSourceConfig(uris=[img_path], channel_order=[0])
    b_source = RasterioSourceConfig(uris=[img_path], channel_order=[0])

    cfg = MultiRasterSourceConfig(
        raster_sources=[
            SubRasterSourceConfig(raster_source=r_source, target_channels=[0]),
            SubRasterSourceConfig(raster_source=g_source, target_channels=[1]),
            SubRasterSourceConfig(raster_source=b_source, target_channels=[2])
        ],
        **kwargs)
    return cfg


class TestMultiRasterSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = rv_config.get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_extent(self):
        cfg = make_cfg('small-rgb-tile.tif')
        rs = cfg.build(tmp_dir=self.tmp_dir)
        extent = rs.get_extent()
        h, w = extent.get_height(), extent.get_width()
        ymin, xmin, ymax, xmax = extent
        self.assertEqual(h, 256)
        self.assertEqual(w, 256)
        self.assertEqual(ymin, 0)
        self.assertEqual(xmin, 0)
        self.assertEqual(ymax, 256)
        self.assertEqual(xmax, 256)

    def test_extent_crop(self):
        f = 1 / 4
        cfg_crop = make_cfg('small-rgb-tile.tif', extent_crop=(f, f, f, f))
        rs_crop = cfg_crop.build(tmp_dir=self.tmp_dir)

        # test extent box
        extent_crop = rs_crop.get_extent()
        self.assertEqual(extent_crop.ymin, 64)
        self.assertEqual(extent_crop.xmin, 64)
        self.assertEqual(extent_crop.ymax, 192)
        self.assertEqual(extent_crop.xmax, 192)

        # test windows
        windows = extent_crop.get_windows(64, 64)
        self.assertEqual(windows[0].ymin, 64)
        self.assertEqual(windows[0].xmin, 64)
        self.assertEqual(windows[-1].ymax, 192)
        self.assertEqual(windows[-1].xmax, 192)

        # test CropOffsets class
        cfg_crop = make_cfg(
            'small-rgb-tile.tif',
            extent_crop=CropOffsets(skip_top=.5, skip_right=.5))
        rs_crop = cfg_crop.build(tmp_dir=self.tmp_dir)
        extent_crop = rs_crop.get_extent()

        self.assertEqual(extent_crop.ymin, 128)
        self.assertEqual(extent_crop.xmin, 0)
        self.assertEqual(extent_crop.ymax, 256)
        self.assertEqual(extent_crop.xmax, 128)

        # test validation
        extent_crop = CropOffsets(skip_top=.5, skip_bottom=.5)
        self.assertRaises(
            ValidationError,
            lambda: make_cfg('small-rgb-tile.tif', extent_crop=extent_crop))

        extent_crop = CropOffsets(skip_left=.5, skip_right=.5)
        self.assertRaises(
            ValidationError,
            lambda: make_cfg('small-rgb-tile.tif', extent_crop=extent_crop))

        # test extent_crop=None
        try:
            _ = make_cfg('small-rgb-tile.tif', extent_crop=None)  # noqa
        except Exception:
            self.fail('extent_crop=None caused an error.')

    def test_nonidentical_extents_and_resolutions(self):
        img_path_1 = data_file_path(
            'multi_raster_source/const_100_600x600.tiff')
        img_path_2 = data_file_path('multi_raster_source/const_175_60x60.tiff')
        img_path_3 = data_file_path('multi_raster_source/const_250_6x6.tiff')
        source_1 = RasterioSourceConfig(uris=[img_path_1], channel_order=[0])
        source_2 = RasterioSourceConfig(uris=[img_path_2], channel_order=[0])
        source_3 = RasterioSourceConfig(uris=[img_path_3], channel_order=[0])

        cfg = MultiRasterSourceConfig(raster_sources=[
            SubRasterSourceConfig(raster_source=source_1, target_channels=[0]),
            SubRasterSourceConfig(raster_source=source_2, target_channels=[1]),
            SubRasterSourceConfig(raster_source=source_3, target_channels=[2])
        ])
        rs = cfg.build(tmp_dir=self.tmp_dir)
        with rs.activate():
            ch_1_only = rs.get_chip(Box(0, 0, 10, 10))
            self.assertEqual(
                tuple(ch_1_only.reshape(-1, 3).mean(axis=0)), (100, 0, 0))
            ch_2_only = rs.get_chip(Box(0, 600, 10, 600 + 10))
            self.assertEqual(
                tuple(ch_2_only.reshape(-1, 3).mean(axis=0)), (0, 175, 0))
            ch_3_only = rs.get_chip(Box(600, 0, 600 + 10, 10))
            self.assertEqual(
                tuple(ch_3_only.reshape(-1, 3).mean(axis=0)), (0, 0, 250))
            full_img = rs.get_chip(Box(0, 0, 600, 600))
            self.assertEqual(set(np.unique(full_img[..., 0])), {100})
            self.assertEqual(set(np.unique(full_img[..., 1])), {0, 175})
            self.assertEqual(set(np.unique(full_img[..., 2])), {0, 250})


if __name__ == '__main__':
    unittest.main()
