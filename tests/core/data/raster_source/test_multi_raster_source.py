from typing import Callable
import unittest

import numpy as np
from xarray import DataArray

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import (
    RasterioSourceConfig, MultiRasterSource, MultiRasterSourceConfig,
    ReclassTransformerConfig, CastTransformerConfig, XarraySource,
    IdentityCRSTransformer, TemporalMultiRasterSource)

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


class TestMultiRasterSourceConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_validate_primary_source_idx(self):
        with self.assertRaises(IndexError):
            _ = make_cfg(primary_source_idx=10)
        self.assertNoError(lambda: make_cfg(primary_source_idx=0))
        self.assertNoError(lambda: make_cfg(primary_source_idx=1))
        self.assertNoError(lambda: make_cfg(primary_source_idx=2))

    def test_validate_temporal(self):
        with self.assertRaises(ValueError):
            _ = make_cfg(temporal=True, channel_order=[0, 1, 2])
        self.assertNoError(lambda: make_cfg(temporal=True))

    def test_build(self):
        cfg = make_cfg()
        self.assertNoError(lambda: cfg.build(tmp_dir=get_tmp_dir()))

    def test_build_with_bbox(self):
        cfg = make_cfg(bbox=(0, 0, 1, 1))
        rs = cfg.build(tmp_dir=get_tmp_dir())
        self.assertEqual(rs.bbox, Box(0, 0, 1, 1))

    def test_build_temporal(self):
        cfg = make_cfg(temporal=True)
        rs = cfg.build(tmp_dir=get_tmp_dir())
        self.assertIsInstance(rs, TemporalMultiRasterSource)
        self.assertEqual(rs.shape, (3, 256, 256, 3))


class TestMultiRasterSource(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = get_tmp_dir()
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

        self.assertEqual(rs.crs_transformer.transform,
                         primary_rs.crs_transformer.transform)
        self.assertNotEqual(rs.crs_transformer, non_primary_rs.crs_transformer)

    def test_bbox(self):
        # /wo user specified extent
        cfg = make_cfg('small-rgb-tile.tif')
        rs = cfg.build(tmp_dir=self.tmp_dir)
        self.assertEqual(rs.bbox, Box(0, 0, 256, 256))
        self.assertEqual(rs.extent, Box(0, 0, 256, 256))

        # /w user specified extent
        cfg_crop = make_cfg('small-rgb-tile.tif', bbox=(64, 64, 192, 192))
        rs_crop = cfg_crop.build(tmp_dir=self.tmp_dir)

        # test extent box
        self.assertEqual(rs_crop.bbox, Box(64, 64, 192, 192))
        self.assertEqual(rs_crop.extent, Box(0, 0, 128, 128))

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
        chip = rs._get_chip(window)
        self.assertEqual(
            tuple(chip.reshape(-1, 3).mean(axis=0)), (100, 175, 250))

        # sub transformers, channel_order, and transformer applied
        chip = rs.get_chip(window)
        self.assertEqual(tuple(chip.reshape(-1, 3).mean(axis=0)), (25, 17, 10))

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

    def test_temporal_sub_raster_sources(self):
        dtype = np.uint8
        arr = np.ones((2, 5, 5, 4), dtype=dtype)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['time', 'x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer(), temporal=True)
        mrs = MultiRasterSource([rs, rs])
        self.assertEqual(mrs.shape, (2, 5, 5, 8))

        chip = mrs.get_chip(Box(0, 0, 2, 2))
        chip_expected = np.ones((2, 2, 2, 8), dtype=dtype)
        chip_expected[..., :4] *= np.arange(4, dtype=np.uint8)
        chip_expected[..., 4:] *= np.arange(4, dtype=np.uint8)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs.get_chip(Box(0, 0, 2, 2), out_shape=(1, 1))
        chip_expected = np.ones((2, 1, 1, 8), dtype=dtype)
        chip_expected[..., :4] *= np.arange(4, dtype=np.uint8)
        chip_expected[..., 4:] *= np.arange(4, dtype=np.uint8)
        np.testing.assert_array_equal(chip, chip_expected)


if __name__ == '__main__':
    unittest.main()
