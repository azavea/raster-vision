import unittest

import numpy as np

from rastervision.core.raster_transformer import RasterTransformer
from rastervision.core.raster_stats import RasterStats


class TestRasterTransformer(unittest.TestCase):
    def test_no_channel_order_no_stats(self):
        transformer = RasterTransformer()
        chip = np.ones((2, 2, 3)).astype(np.uint8)
        out_chip = transformer.transform(chip)
        np.testing.assert_equal(chip, out_chip)

        # Need to supply raster_stats for non-uint8 chips.
        chip = np.ones((2, 2, 3))
        with self.assertRaises(ValueError):
            out_chip = transformer.transform(chip)

    def test_no_channel_order_has_stats(self):
        raster_stats = RasterStats()
        raster_stats.means = np.ones((4, ))
        raster_stats.stds = np.ones((4, )) * 2

        # All values have z-score of 1, which translates to
        # uint8 value of 170.
        transformer = RasterTransformer(raster_stats=raster_stats)
        chip = np.ones((2, 2, 4)) * 3
        out_chip = transformer.transform(chip)
        expected_out_chip = np.ones((2, 2, 4)) * 170
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_has_channel_order_no_stats(self):
        channel_order = [0, 1, 2]
        transformer = RasterTransformer(channel_order=channel_order)
        chip = np.ones((2, 2, 4)).astype(np.uint8)
        chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
        out_chip = transformer.transform(chip)
        expected_out_chip = np.ones((2, 2, 3)).astype(np.uint8)
        expected_out_chip[:, :, :] *= np.array([0, 1, 2]).astype(np.uint8)
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_has_channel_order_has_stats(self):
        raster_stats = RasterStats()
        raster_stats.means = np.ones((4, ))
        raster_stats.stds = np.ones((4, )) * 2
        channel_order = [0, 1, 2]
        transformer = RasterTransformer(
            raster_stats=raster_stats, channel_order=channel_order)

        chip = np.ones((2, 2, 4)) * [3, 3, 3, 0]
        out_chip = transformer.transform(chip)
        expected_out_chip = np.ones((2, 2, 3)) * 170
        np.testing.assert_equal(out_chip, expected_out_chip)

        # Also test when chip has same number of channels as channel_order
        # but different number of channels than stats.
        chip = np.ones((2, 2, 3)) * [3, 3, 3]
        out_chip = transformer.transform(chip)
        expected_out_chip = np.ones((2, 2, 3)) * 170
        np.testing.assert_equal(out_chip, expected_out_chip)


if __name__ == '__main__':
    unittest.main()
