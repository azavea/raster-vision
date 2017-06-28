import unittest

import numpy as np

from rastervision.tagging.data.planet_kaggle import Dataset, TagStore
from rastervision.tagging.tasks.predict import compute_prediction


class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.tag_store = TagStore(active_tags=self.dataset.all_tags)
        self.nb_tags = len(self.tag_store.active_tags)

    def test_compute_predictions1(self):
        y_probs = np.zeros((self.nb_tags,))
        haze_ind = self.tag_store.get_tag_ind(self.dataset.haze)
        road_ind = self.tag_store.get_tag_ind(self.dataset.road)
        y_probs[haze_ind] = 0.5
        y_probs[road_ind] = 0.7

        thresholds = np.zeros((self.nb_tags,))
        thresholds[haze_ind] = 0.4
        thresholds[road_ind] = 0.6

        y_pred = compute_prediction(
            y_probs, self.dataset, self.tag_store, thresholds)
        y = np.zeros((self.nb_tags,))
        y[[haze_ind, road_ind]] = 1
        self.assertTrue(np.array_equal(y_pred, y))

    def test_compute_predictions2(self):
        y_probs = np.zeros((self.nb_tags,))
        haze_ind = self.tag_store.get_tag_ind(self.dataset.haze)
        road_ind = self.tag_store.get_tag_ind(self.dataset.road)
        y_probs[haze_ind] = 0.1
        y_probs[road_ind] = 0.1

        thresholds = np.zeros((self.nb_tags,))
        thresholds[haze_ind] = 0.2
        thresholds[road_ind] = 0.2

        y_pred = compute_prediction(
            y_probs, self.dataset, self.tag_store, thresholds)
        y = np.zeros((self.nb_tags,))
        y[[haze_ind]] = 1

        self.assertTrue(np.array_equal(y_pred, y))


if __name__ == '__main__':
    unittest.main()
