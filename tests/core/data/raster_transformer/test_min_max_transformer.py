import unittest

import numpy as np

from rastervision.core.data import MinMaxTransformer, MinMaxTransformerConfig


class TestMinMaxTransformerConfig(unittest.TestCase):
    def test_build(self):
        cfg = MinMaxTransformerConfig()
        tf = cfg.build()
        self.assertIsInstance(tf, MinMaxTransformer)


class TestMinMaxTransformer(unittest.TestCase):
    def test_transform(self):
        tf = MinMaxTransformer()
        chip_in = np.array([
            [[-2], [0]],
            [[8], [18]],
        ])
        chip_expexted = np.array(
            [
                [[0], [25]],
                [[127], [255]],
            ], dtype=np.uint8)
        chip_out = tf.transform(chip_in)
        np.testing.assert_array_equal(chip_out, chip_expexted)

        # temporal data
        chip_in = chip_in[np.newaxis]
        chip_expexted = chip_expexted[np.newaxis]
        chip_out = tf.transform(chip_in)
        np.testing.assert_array_equal(chip_out, chip_expexted)


if __name__ == '__main__':
    unittest.main()
