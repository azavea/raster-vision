import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import (
    RasterioSourceConfig, MultiRasterSourceConfig, ReclassTransformerConfig,
    CastTransformerConfig)
from rastervision.pipeline import rv_config

from tests import data_file_path


def make_cfg(img_path: str = 'small-rgb-tile.tif',
             **kwargs) -> MultiRasterSourceConfig:
    img_path = data_file_path(img_path)
    r_source = RasterioSourceConfig(uris=[img_path])
    g_source = RasterioSourceConfig(uris=[img_path])
    b_source = RasterioSourceConfig(uris=[img_path])

    cfg = MultiRasterSourceConfig(
        raster_sources=[r_source, g_source, b_source], **kwargs)
    return cfg


def make_cfg_diverse(diff_dtypes: bool = False,
                     **kwargs) -> MultiRasterSourceConfig:
    img_paths = [
        data_file_path('multi_raster_source/const_100_600x600.tiff'),
        data_file_path('multi_raster_source/const_175_60x60.tiff'),
        data_file_path('multi_raster_source/const_250_6x6.tiff')
    ]
    transformers = [[]] * 3
    if diff_dtypes:
        transformers = [
            [],
            [CastTransformerConfig(to_dtype='float32')],
            [CastTransformerConfig(to_dtype='int')],
        ]
    rs_cfgs = [
        RasterioSourceConfig(uris=[path], transformers=tfs)
        for path, tfs in zip(img_paths, transformers)
    ]
    cfg = MultiRasterSourceConfig(raster_sources=rs_cfgs, **kwargs)
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
        extent = rs.extent
        h, w = extent.size
        self.assertEqual(h, 256)
        self.assertEqual(w, 256)
        ymin, xmin, ymax, xmax = extent
        self.assertEqual(ymin, 0)
        self.assertEqual(xmin, 0)
        self.assertEqual(ymax, 256)
        self.assertEqual(xmax, 256)

    def test_primary_source_idx(self):
        primary_source_idx = 2
        non_primary_source_idx = 1

        cfg = make_cfg_diverse(
            diff_dtypes=True,
            force_same_dtype=True,
            primary_source_idx=primary_source_idx)
        rs = cfg.build(tmp_dir=self.tmp_dir)
        primary_rs = rs.raster_sources[primary_source_idx]
        non_primary_rs = rs.raster_sources[non_primary_source_idx]

        self.assertEqual(rs.extent, primary_rs.extent)
        self.assertNotEqual(rs.extent, non_primary_rs.extent)

        self.assertEqual(rs.dtype, primary_rs.dtype)
        self.assertNotEqual(rs.dtype, non_primary_rs.dtype)

        self.assertEqual(rs.get_crs_transformer().transform,
                         primary_rs.get_crs_transformer().transform)
        self.assertNotEqual(rs.get_crs_transformer(),
                            non_primary_rs.get_crs_transformer())

    def test_user_specified_extent(self):
        # /wo user specified extent
        cfg = make_cfg('small-rgb-tile.tif')
        rs = cfg.build(tmp_dir=self.tmp_dir)
        self.assertEqual(rs.extent, Box(0, 0, 256, 256))

        # test validators
        cfg = make_cfg('small-rgb-tile.tif', extent=(64, 64, 192, 192))
        self.assertIsInstance(cfg.extent, Box)

        # /w user specified extent
        cfg_crop = make_cfg('small-rgb-tile.tif', extent=(64, 64, 192, 192))
        rs_crop = cfg_crop.build(tmp_dir=self.tmp_dir)

        # test extent box
        self.assertEqual(rs_crop.extent, Box(64, 64, 192, 192))

    def test_get_chip(self):
        # create a 3-channel raster from a 1-channel raster
        # using a ReclassTransformer to give each channel a different value
        # (100, 175, and 250 respectively)
        path = data_file_path('multi_raster_source/const_100_600x600.tiff')
        source_1 = RasterioSourceConfig(uris=[path], channel_order=[0])
        source_2 = RasterioSourceConfig(
            uris=[path],
            channel_order=[0],
            transformers=[ReclassTransformerConfig(mapping={100: 175})])
        source_3 = RasterioSourceConfig(
            uris=[path],
            channel_order=[0],
            transformers=[ReclassTransformerConfig(mapping={100: 250})])

        cfg = MultiRasterSourceConfig(
            raster_sources=[source_1, source_2, source_3],
            channel_order=[2, 1, 0],
            transformers=[
                ReclassTransformerConfig(mapping={
                    100: 10,
                    175: 17,
                    250: 25
                })
            ])
        rs = cfg.build(tmp_dir=self.tmp_dir)

        window = Box(0, 0, 100, 100)

        # sub transformers and channel_order applied
        sub_chips = rs._get_sub_chips(window, raw=False)
        self.assertEqual(tuple(c.mean() for c in sub_chips), (100, 175, 250))
        # sub transformers, channel_order, and transformer applied
        chip = rs.get_chip(window)
        self.assertEqual(tuple(chip.reshape(-1, 3).mean(axis=0)), (25, 17, 10))

        # none of sub transformers, channel_order, and transformer applied
        sub_chips = rs._get_sub_chips(window, raw=True)
        self.assertEqual(tuple(c.mean() for c in sub_chips), (100, 100, 100))
        chip = rs._get_chip(window)
        self.assertEqual(
            tuple(chip.reshape(-1, 3).mean(axis=0)), (100, 100, 100))

    def test_nonidentical_extents_and_resolutions(self):
        cfg = make_cfg_diverse(diff_dtypes=False)
        rs = cfg.build(tmp_dir=self.tmp_dir)
        for get_chip_fn in [rs._get_chip, rs.get_chip]:
            ch_1_only = get_chip_fn(Box(0, 0, 10, 10))
            self.assertEqual(
                tuple(ch_1_only.reshape(-1, 3).mean(axis=0)), (100, 0, 0))
            ch_2_only = get_chip_fn(Box(0, 600, 10, 600 + 10))
            self.assertEqual(
                tuple(ch_2_only.reshape(-1, 3).mean(axis=0)), (0, 175, 0))
            ch_3_only = get_chip_fn(Box(600, 0, 600 + 10, 10))
            self.assertEqual(
                tuple(ch_3_only.reshape(-1, 3).mean(axis=0)), (0, 0, 250))
            full_img = get_chip_fn(Box(0, 0, 600, 600))
            self.assertEqual(set(np.unique(full_img[..., 0])), {100})
            self.assertEqual(set(np.unique(full_img[..., 1])), {0, 175})
            self.assertEqual(set(np.unique(full_img[..., 2])), {0, 250})


if __name__ == '__main__':
    unittest.main()
