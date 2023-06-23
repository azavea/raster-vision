from typing import Callable, List
import unittest

import numpy as np
from xarray import DataArray

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import IdentityCRSTransformer
from rastervision.core.data.raster_source import (TemporalMultiRasterSource,
                                                  XarraySource)


def make_raster_source(num_channels_raw: int, channel_order: List[int]):
    dtype = np.uint8
    arr = np.ones((5, 5, num_channels_raw), dtype=dtype)
    arr *= np.arange(num_channels_raw, dtype=dtype)
    da = DataArray(arr, dims=['x', 'y', 'band'])
    rs = XarraySource(
        da, IdentityCRSTransformer(), channel_order=channel_order)
    return rs


def make_sub_raster_sources():
    rs1 = make_raster_source(4, [2, 1, 0])
    rs2 = make_raster_source(4, [1, 0, 2])
    return [rs1, rs2]


def make_source():
    mrs = TemporalMultiRasterSource(make_sub_raster_sources())
    return mrs


class TestTemporalMultiRasterSource(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_init_temporal(self):
        rs1 = make_raster_source(3, [0, 1, 2])
        rs2 = make_raster_source(4, [0, 1, 2, 3])
        rs3 = make_raster_source(4, [0, 1, 2])
        self.assertRaises(ValueError,
                          lambda: TemporalMultiRasterSource([rs1, rs2]))
        self.assertNoError(lambda: TemporalMultiRasterSource([rs1, rs3]))

        args = dict(raster_sources=[rs1, rs3], primary_source_idx=10)
        self.assertRaises(IndexError,
                          lambda: TemporalMultiRasterSource(**args))

    def test_get_chip(self):
        mrs = make_source()
        dtype = mrs.dtype

        chip = mrs.get_chip(Box(0, 0, 1, 1))
        chip_expected = np.array(
            [
                [[[2, 1, 0]]],
                [[[1, 0, 2]]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs.get_chip(Box(0, 0, 2, 2), out_shape=(1, 1))
        chip_expected = np.array(
            [
                [[[2, 1, 0]]],
                [[[1, 0, 2]]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_getitem(self):
        mrs = make_source()
        dtype = mrs.dtype

        chip = mrs[Box(0, 0, 1, 1)]
        chip_expected = np.array(
            [
                [[[2, 1, 0]]],
                [[[1, 0, 2]]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs[:, :1, :1, [2]]
        chip_expected = np.array(
            [
                [[[0]]],
                [[[2]]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs[:, :2:2, :2:2, [0]]
        chip_expected = np.array(
            [
                [[[2]]],
                [[[1]]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = mrs[1, :2:2, :2:2, [0]]
        chip_expected = np.array(
            [
                [[1]],
            ], dtype=dtype)
        np.testing.assert_array_equal(chip, chip_expected)


if __name__ == '__main__':
    unittest.main()
