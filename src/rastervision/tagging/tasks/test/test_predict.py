import unittest

import numpy as np

from rastervision.tagging.data.planet_kaggle import Dataset
from rastervision.tagging.tasks.predict import compute_prediction


class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()

    def test_compute_predictions1(self):
        y_probs = np.zeros((self.dataset.nb_tags,))
        haze_ind = self.dataset.get_tag_ind(self.dataset.haze)
        road_ind = self.dataset.get_tag_ind(self.dataset.road)
        y_probs[haze_ind] = 0.6
        y_probs[road_ind] = 0.7

        y_pred = compute_prediction(y_probs, self.dataset)
        y = np.zeros((self.dataset.nb_tags,))
        y[[haze_ind, road_ind]] = 1

        self.assertTrue(np.array_equal(y_pred, y))

    def test_compute_predictions2(self):
        y_probs = np.zeros((self.dataset.nb_tags,))
        haze_ind = self.dataset.get_tag_ind(self.dataset.haze)
        road_ind = self.dataset.get_tag_ind(self.dataset.road)
        y_probs[haze_ind] = 0.1
        y_probs[road_ind] = 0.1

        y_pred = compute_prediction(y_probs, self.dataset)
        y = np.zeros((self.dataset.nb_tags,))
        y[[haze_ind]] = 1

        self.assertTrue(np.array_equal(y_pred, y))


if __name__ == '__main__':
    unittest.main()
