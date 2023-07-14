import unittest
from os.path import join

import numpy as np
from xarray import DataArray

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import IdentityCRSTransformer, XarraySource
from rastervision.core.raster_stats import (
    RasterStats, get_num_chips_to_sample, random_chip_stream,
    sliding_chip_stream, get_chip, parallel_mean, parallel_variance)


class TestRasterStats(unittest.TestCase):
    def test_save_and_load(self):
        stats = RasterStats(np.array([1, 2]), np.array([3, 4]))
        with get_tmp_dir() as tmp_dir:
            stats_uri = join(tmp_dir, 'stats.json')
            stats.save(stats_uri)

            stats2 = RasterStats.load(stats_uri)
            np.testing.assert_array_equal(stats2.means, np.array([1, 2]))
            np.testing.assert_array_equal(stats2.stds, np.array([3, 4]))

    def test_no_valid_chips(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        stats = RasterStats()
        args = dict(raster_sources=[rs], chip_sz=10, stride=10)
        self.assertRaises(ValueError, lambda: stats.compute(**args))
        args = dict(raster_sources=[rs], chip_sz=10, sample_prob=1)
        self.assertRaises(ValueError, lambda: stats.compute(**args))

    def test_compute_from_pixels_validation(self):
        stats = RasterStats()
        pixels = np.zeros((5, 3), dtype=np.uint8)
        running_mean = np.zeros((3, ), dtype=np.uint8)
        args = dict(pixels=pixels, running_mean=running_mean)
        self.assertRaises(ValueError,
                          lambda: stats.compute_from_pixels(**args))


class TestUtils(unittest.TestCase):
    def test_parallel_mean(self):
        a = np.random.randint(0, 10, size=5)
        b = np.random.randint(0, 10, size=10)
        mean = parallel_mean(a.mean(), len(a), b.mean(), len(b))
        expected_mean = np.concatenate((a, b)).mean()
        self.assertEqual(mean, expected_mean)

    def test_parallel_variance(self):
        a = np.random.randint(0, 10, size=5)
        b = np.random.randint(0, 10, size=10)
        var = parallel_variance(
            a.mean(), len(a), a.var(ddof=1), b.mean(), len(b), b.var(ddof=1))
        expected_var = np.concatenate((a, b)).var(ddof=1)
        self.assertAlmostEqual(var, expected_var)

    def test_get_num_chips_to_sample(self):
        n = get_num_chips_to_sample(
            extent=Box(0, 0, 1, 1), chip_sz=10, sample_prob=0.1)
        self.assertEqual(n, 0)
        n = get_num_chips_to_sample(
            extent=Box(0, 0, 100, 100), chip_sz=10, sample_prob=0.)
        self.assertEqual(n, 1)
        n = get_num_chips_to_sample(
            extent=Box(0, 0, 100, 100), chip_sz=10, sample_prob=0.1)
        self.assertEqual(n, 10)

    def test_get_chip(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        arr[:10, :10] = 1
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chip = get_chip(rs, Box(0, 0, 10, 10), nodata_value=0)
        self.assertIsNotNone(chip)
        chip = get_chip(rs, Box(9, 9, 20, 20), nodata_value=0)
        self.assertIsNotNone(chip)
        chip = get_chip(rs, Box(10, 10, 20, 20), nodata_value=0)
        self.assertIsNone(chip)
        chip = get_chip(rs, Box(10, 10, 20, 20), nodata_value=None)
        self.assertIsNotNone(chip)

    def test_sliding_chip_stream_normal(self):
        arr = np.ones((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chips = list(sliding_chip_stream([rs], chip_sz=10, stride=10))
        self.assertEqual(len(chips), 4)

    def test_sliding_chip_stream_all_nodata(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chips = list(sliding_chip_stream([rs], chip_sz=10, stride=10))
        self.assertEqual(len(chips), 0)

    def test_random_chip_stream_normal(self):
        arr = np.ones((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chips = list(random_chip_stream([rs], chip_sz=10, sample_prob=0.5))
        self.assertEqual(len(chips), 2)
        chips = list(random_chip_stream([rs], chip_sz=10, sample_prob=0.))
        self.assertEqual(len(chips), 1)

    def test_random_chip_stream_all_nodata(self):
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chips = list(random_chip_stream([rs], chip_sz=10, sample_prob=0.5))
        self.assertEqual(len(chips), 0)

    def test_random_chip_stream_extent_smaller_than_window(self):
        arr = np.ones((20, 20, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        chips = list(random_chip_stream([rs], chip_sz=100, sample_prob=0.5))
        self.assertEqual(len(chips), 1)
        self.assertEqual(chips[0].shape, (20, 20, 4))

    def test_random_chip_stream_window_overflows_extent(self):
        arr = np.ones((20, 100, 4), dtype=np.uint8)
        da = DataArray(arr, dims=['x', 'y', 'band'])
        rs = XarraySource(da, IdentityCRSTransformer())
        args = dict(raster_sources=[rs], chip_sz=40, sample_prob=0.5)
        self.assertRaises(ValueError, lambda: list(random_chip_stream(**args)))


if __name__ == '__main__':
    unittest.main()
