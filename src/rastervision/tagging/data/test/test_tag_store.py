import unittest

import numpy as np

from rastervision.tagging.data.planet_kaggle import TagStore, Dataset


class TagStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.active_tags = [
            self.dataset.bare_ground, self.dataset.partly_cloudy,
            self.dataset.artisinal_mine]
        self.tag_store = TagStore(active_tags=self.active_tags)

        self.str_tags = [self.dataset.bare_ground, self.dataset.partly_cloudy]
        self.binary_tags = np.zeros((len(self.active_tags),))
        self.binary_tags[[
            self.tag_store.get_tag_ind(self.dataset.bare_ground),
            self.tag_store.get_tag_ind(self.dataset.partly_cloudy)]] = 1

        self.ind1 = 'ind1'
        self.ind2 = 'ind2'
        self.tag_store.add_csv_row((self.ind1, 'bare_ground partly_cloudy'))
        self.tag_store.add_csv_row((self.ind2, 'bare_ground'))

    def test_strs_to_binary(self):
        # Should ignore blooming because it's not in active_tags.
        str_tags = [
            self.dataset.bare_ground, self.dataset.partly_cloudy,
            self.dataset.blooming]
        self.assertTrue(np.array_equal(
            self.tag_store.strs_to_binary(str_tags),
            self.binary_tags))

    def test_binary_to_str(self):
        self.assertEqual(
            self.tag_store.binary_to_strs(self.binary_tags),
            self.str_tags)

    def test_get_tag_array(self):
        tag_array = self.tag_store.get_tag_array([self.ind1, self.ind2])
        ind1_tags = self.tag_store.binary_to_strs(tag_array[0, :])
        ind2_tags = self.tag_store.binary_to_strs(tag_array[1, :])
        self.assertEqual(set(ind1_tags), set(['bare_ground', 'partly_cloudy']))
        self.assertEqual(set(ind2_tags), set(['bare_ground']))

    def test_get_tag_counts(self):
        tag_counts = self.tag_store.get_tag_counts()
        self.assertEqual(len(tag_counts), len(self.active_tags))
        self.assertEqual(tag_counts['bare_ground'], 2)
        self.assertEqual(tag_counts['partly_cloudy'], 1)
        count_sum = sum(tag_counts.values())
        self.assertEqual(count_sum, 3)

    def test_get_tag_diff(self):
        y_true = self.tag_store.strs_to_binary(
            [self.dataset.bare_ground, self.dataset.partly_cloudy])
        y_pred = self.tag_store.strs_to_binary(
            [self.dataset.bare_ground, self.dataset.artisinal_mine])

        add_pred_tags, remove_pred_tags = \
            self.tag_store.get_tag_diff(y_true, y_pred)
        self.assertEqual(add_pred_tags, [self.dataset.artisinal_mine])
        self.assertEqual(remove_pred_tags, [self.dataset.partly_cloudy])

    def test_compute_train_probs(self):
        ind3 = 'ind3'
        self.tag_store.add_csv_row((ind3, 'blooming'))
        active_tags_prob = 0.5
        sample_probs = \
            self.tag_store.compute_sample_probs(
                [self.ind1, self.ind2, ind3], active_tags_prob)
        self.assertTrue(np.array_equal(sample_probs, [0.25, 0.25, 0.5]))


if __name__ == '__main__':
    unittest.main()
