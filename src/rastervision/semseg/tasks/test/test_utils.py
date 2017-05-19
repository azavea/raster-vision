import unittest

import numpy as np

from rastervision.semseg.tasks.utils import make_prediction_img


class MakePredictionimgTestCase(unittest.TestCase):
    def test_identity_predict(self):
        # When predict is the identity function, the input and output
        # should be equal.
        full_img = np.random.randint(0, 2, size=(200, 200, 1))
        target_size = 64

        def predict(img):
            return img

        output_img = make_prediction_img(
            full_img, target_size, predict)
        self.assertTrue(np.array_equal(full_img, output_img))


if __name__ == '__main__':
    unittest.main()
