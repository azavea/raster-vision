import unittest

import numpy as np

from rastervision.core.rv_pipeline.utils import nodata_below_threshold


class TestUtils(unittest.TestCase):
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

    def test_nodata_below_threshold_temporal(self):
        chip = np.ones((2, 256, 256, 3), dtype=np.uint8)

        # different proportions of nodata
        mask_0 = np.ones_like(chip)
        mask_50 = np.ones_like(chip)
        mask_50[:, :128] = 0
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
