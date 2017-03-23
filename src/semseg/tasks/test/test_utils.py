import unittest

import numpy as np

from ..utils import make_prediction_tile


class MakePredictionTileTestCase(unittest.TestCase):
    def test_identity_predict(self):
        # When predict is the identity function, the input and output
        # should be equal.
        full_tile = np.random.randint(0, 2, size=(200, 200, 1))
        tile_size = 64

        def predict(tile):
            return tile

        output_tile = make_prediction_tile(
            full_tile, tile_size, predict)
        self.assertTrue(np.array_equal(full_tile, output_tile))


if __name__ == '__main__':
    unittest.main()
