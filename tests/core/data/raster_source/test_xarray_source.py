import unittest

import numpy as np
from xarray import DataArray

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import IdentityCRSTransformer
from rastervision.core.data.raster_source import (ChannelOrderError,
                                                  XarraySource)


class TestXarraySource(unittest.TestCase):
    def test_init_temporal(self):
        arr = np.ones((2, 5, 5, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['time', 'x', 'y', 'band'])
        with self.assertRaises(ValueError):
            _ = XarraySource(
                da, IdentityCRSTransformer(), channel_order=[2, 1, 0])

    def test_dtype(self):
        dtype = np.uint8
        arr = np.empty((5, 5, 3), dtype=dtype)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertEqual(rs.dtype, dtype)

        dtype = np.float32
        arr = np.empty((5, 5, 3), dtype=dtype)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertEqual(rs.dtype, dtype)

    def test_num_channels(self):
        num_channels = 3
        arr = np.empty((5, 5, num_channels), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertEqual(rs.num_channels, num_channels)

        num_channels = 8
        arr = np.empty((5, 5, num_channels), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertEqual(rs.num_channels, num_channels)

    def test_crs_transformer(self):
        arr = np.empty((5, 5, 3), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertIsInstance(rs.crs_transformer, IdentityCRSTransformer)

    def test_get_raw_chip(self):
        arr = np.ones((5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(
            da, IdentityCRSTransformer(), channel_order=[2, 1, 0])
        chip = rs.get_raw_chip(Box(0, 0, 1, 1))
        chip_expected = np.array([[[0, 1, 2, 3]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_get_chip(self):
        arr = np.ones((5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(
            da, IdentityCRSTransformer(), channel_order=[2, 1, 0])
        chip = rs.get_chip(Box(0, 0, 1, 1))
        chip_expected = np.array([[[2, 1, 0]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs.get_chip(Box(0, 0, 1, 1), bands=[2])
        chip_expected = np.array([[[0]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs.get_chip(Box(0, 0, 2, 2), bands=[0], out_shape=(1, 1))
        chip_expected = np.array([[[2]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_get_chip_temporal(self):
        arr = np.ones((2, 5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        arr[1] *= 2
        da = DataArray(arr, dims=['time', 'x', 'y', 'band'])
        rs = XarraySource(
            da,
            IdentityCRSTransformer(),
            channel_order=[2, 1, 0],
            temporal=True)
        chip = rs.get_chip(Box(0, 0, 1, 1))
        chip_expected = np.array(
            [
                [[[2, 1, 0]]],
                [[[4, 2, 0]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs.get_chip(Box(0, 0, 1, 1), bands=[2])
        chip_expected = np.array(
            [
                [[[0]]],
                [[[0]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs.get_chip(Box(0, 0, 2, 2), bands=[0], out_shape=(1, 1))
        chip_expected = np.array(
            [
                [[[2]]],
                [[[4]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs.get_chip(
            Box(0, 0, 2, 2), bands=[0], time=1, out_shape=(1, 1))
        chip_expected = np.array(
            [
                [[4]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_getitem(self):
        arr = np.ones((5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(
            da, IdentityCRSTransformer(), channel_order=[2, 1, 0])
        chip = rs[Box(0, 0, 1, 1)]
        chip_expected = np.array([[[2, 1, 0]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs[:1, :1, [2]]
        chip_expected = np.array([[[0]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs[:2:2, :2:2, [0]]
        chip_expected = np.array([[[2]]], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_getitem_temporal(self):
        arr = np.ones((2, 5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        arr[1] *= 2
        da = DataArray(arr, dims=['time', 'x', 'y', 'band'])
        rs = XarraySource(
            da,
            IdentityCRSTransformer(),
            channel_order=[2, 1, 0],
            temporal=True)
        chip = rs[Box(0, 0, 1, 1)]
        chip_expected = np.array(
            [
                [[[2, 1, 0]]],
                [[[4, 2, 0]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs[:, :1, :1, [2]]
        chip_expected = np.array(
            [
                [[[0]]],
                [[[0]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs[:, :2:2, :2:2, [0]]
        chip_expected = np.array(
            [
                [[[2]]],
                [[[4]]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

        chip = rs[1, :2:2, :2:2, [0]]
        chip_expected = np.array(
            [
                [[4]],
            ], dtype=arr.dtype)
        np.testing.assert_array_equal(chip, chip_expected)

    def test_resizing(self):
        arr = np.ones((5, 5, 3), dtype=np.uint8)
        arr *= np.arange(3, dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chip = rs.get_chip(Box(0, 0, 1, 1), out_shape=(5, 5))
        np.testing.assert_array_equal(chip, arr)

    def test_channel_order_error(self):
        arr = np.empty((5, 5, 4), dtype=np.uint8)
        arr *= np.arange(4, dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        with self.assertRaises(ChannelOrderError):
            _ = XarraySource(da, IdentityCRSTransformer(), channel_order=[10])

    def test_extent(self):
        arr = np.empty((5, 5, 3), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        self.assertEqual(rs.extent, Box(0, 0, 5, 5))

    def test_bbox(self):
        arr = np.empty((10, 10, 3), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer(), bbox=Box(2, 2, 8, 8))
        self.assertEqual(rs.bbox, Box(2, 2, 8, 8))
        self.assertEqual(rs.extent, Box(0, 0, 6, 6))

    def test_extent_overflow(self):
        # w/o bbox
        arr = np.ones((5, 5, 3), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        window = Box(0, 0, 5, 10)
        chip = rs.get_chip(window)
        chip_expected = np.ones((*window.size, arr.shape[-1]), dtype=arr.dtype)
        chip_expected[:, 5:] = 0
        np.testing.assert_array_equal(chip, chip_expected)

        # w/ bbox
        arr = np.ones((5, 5, 3), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer(), bbox=Box(0, 0, 4, 4))
        window = Box(0, 0, 6, 6)
        chip = rs.get_chip(window)
        chip_expected = np.ones((*window.size, arr.shape[-1]), dtype=arr.dtype)
        chip_expected[:, -2:] = 0
        chip_expected[-2:, :] = 0
        np.testing.assert_array_equal(chip, chip_expected)


if __name__ == '__main__':
    unittest.main()
