import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.utils.raster import (pad_to_window_size,
                                                 fill_overflow)


class TestRasterUtils(unittest.TestCase):
    def test_fill_overflow(self):
        bbox = Box(10, 10, 90, 90)
        window = Box(0, 0, 100, 100)
        chip_in = np.ones((2, 100, 100, 4), dtype=np.uint8)
        chip_out = fill_overflow(bbox, window, chip_in)
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[:, mask] == 1))
        self.assertTrue(np.all(chip_out[:, ~mask] == 0))

    def test_pad_to_window_size(self):
        bbox = Box(10, 10, 90, 90)
        window = Box(0, 0, 100, 100)

        # ndim == 2
        chip_in = np.ones((80, 80), dtype=np.uint8)
        chip_out = pad_to_window_size(chip_in, window, bbox, fill_value=0)
        self.assertEqual(chip_out.shape, (100, 100))
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[mask] == 1))
        self.assertTrue(np.all(chip_out[~mask] == 0))

        # ndim > 2
        chip_in = np.ones((2, 80, 80, 4), dtype=np.uint8)
        chip_out = pad_to_window_size(chip_in, window, bbox, fill_value=0)
        self.assertEqual(chip_out.shape, (2, 100, 100, 4))
        mask = np.zeros((100, 100), dtype=bool)
        mask[10:90, 10:90] = 1
        self.assertTrue(np.all(chip_out[:, mask] == 1))
        self.assertTrue(np.all(chip_out[:, ~mask] == 0))


if __name__ == '__main__':
    unittest.main()
