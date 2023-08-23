import unittest

import numpy as np

from rastervision.core.data import NanTransformer, NanTransformerConfig


class TestNanTransformerConfig(unittest.TestCase):
    def test_build(self):
        cfg = NanTransformerConfig(to_value=1)
        tf = cfg.build()
        self.assertIsInstance(tf, NanTransformer)
        self.assertEqual(tf.to_value, 1)


class TestNanTransformer(unittest.TestCase):
    def test_transform(self):
        tf = NanTransformer(1)
        chip_in = np.array([
            [[np.nan], [0]],
            [[0], [np.nan]],
        ])
        chip_expexted = np.array(
            [
                [[1], [0]],
                [[0], [1]],
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
