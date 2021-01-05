import unittest

import numpy as np

from rastervision.core.data.raster_transformer import CastTransformerConfig


class TestCastTransformer(unittest.TestCase):
    def test_cast_transformer(self):
        in_chip = np.empty((10, 10, 3), dtype=np.float32)
        tf = CastTransformerConfig(to_dtype='uint8').build()
        out_chip = tf.transform(in_chip)
        self.assertEqual(out_chip.dtype, np.uint8)
        self.assertEqual(str(tf), 'CastTransformer(to_dtype="uint8")')

        in_chip = np.empty((10, 10, 3), dtype=np.uint16)
        tf = CastTransformerConfig(to_dtype='float32').build()
        out_chip = tf.transform(in_chip)
        self.assertEqual(out_chip.dtype, np.float32)
        self.assertEqual(str(tf), 'CastTransformer(to_dtype="float32")')


if __name__ == '__main__':
    unittest.main()
