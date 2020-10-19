import unittest

import numpy as np

from rastervision.core.rv_pipeline.utils import (fill_no_data,
                                                 nodata_below_threshold)


class TestUtils(unittest.TestCase):
    def test_fill_no_data(self):
        chip = np.ones((256, 256, 3), dtype=np.uint8)
        label = np.zeros((256, 256), dtype=np.uint8)
        label[:, 128:] = 2

        chip_filled = fill_no_data(chip, label, null_class_id=2)
        self.assertEqual(chip_filled[:, 128:].sum(), 0)
        self.assertEqual(chip_filled[:, :128].sum(), chip[:, :128].sum())

    def test_nodata_below_threshold(self):
        chip = np.ones((256, 256, 3), dtype=np.uint8)
        # different proportions of nodata
        mask_0 = np.ones_like(chip)
        mask_50 = np.ones_like(chip)
        mask_50[:128] = 0
        mask_100 = np.zeros_like(chip)

        # 0% nodata
        test_chip = chip & mask_0
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.00))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=0.25))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=0.50))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=0.75))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=1.00))

        # 50% nodata
        test_chip = chip & mask_50
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.00))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.25))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.50))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=0.75))
        self.assertTrue(nodata_below_threshold(test_chip, threshold=1.00))

        # 100% nodata
        test_chip = chip & mask_100
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.00))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.25))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.50))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=0.75))
        self.assertFalse(nodata_below_threshold(test_chip, threshold=1.00))


if __name__ == '__main__':
    unittest.main()
