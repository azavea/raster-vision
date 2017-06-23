import unittest

import numpy as np

from rastervision.tagging.data.planet_kaggle import TagStore, Dataset


class TagStoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tag_store = TagStore()
        self.dataset = Dataset()
        self.str_tags = [self.dataset.bare_ground, self.dataset.partly_cloudy]
        self.binary_tags = np.zeros((self.dataset.nb_tags,))
        self.binary_tags[[
            self.dataset.get_tag_ind(self.dataset.bare_ground),
            self.dataset.get_tag_ind(self.dataset.partly_cloudy)]] = 1

        self.ind1 = 'ind1'
        self.ind2 = 'ind2'
        self.tag_store.add_csv_row((self.ind1, 'bare_ground partly_cloudy'))
        self.tag_store.add_csv_row((self.ind2, 'bare_ground'))

    def test_strs_to_binary(self):
        self.assertTrue(np.array_equal(
            self.tag_store.strs_to_binary(self.str_tags),
            self.binary_tags))

    def test_binary_to_str(self):
        self.assertEqual(
            self.tag_store.binary_to_strs(self.binary_tags),
            self.str_tags)

    def test_get_tag_array(self):
        tag_array = self.tag_store.get_tag_array([self.ind2, self.ind1])
        ind2_tags = self.tag_store.binary_to_strs(tag_array[0, :])
        ind1_tags = self.tag_store.binary_to_strs(tag_array[1, :])
        self.assertEqual(set(ind2_tags), set(['bare_ground']))
        self.assertEqual(set(ind1_tags), set(['bare_ground', 'partly_cloudy']))

    def test_get_tag_counts(self):
        tag_counts = self.tag_store.get_tag_counts(self.dataset.all_tags)
        self.assertEqual(len(tag_counts), self.dataset.nb_tags)
        self.assertEqual(tag_counts['bare_ground'], 2)
        self.assertEqual(tag_counts['partly_cloudy'], 1)
        count_sum = sum(tag_counts.values())
        self.assertEqual(count_sum, 3)

    def test_get_tag_diff(self):
        y_true = self.tag_store.strs_to_binary(
            [self.dataset.bare_ground, self.dataset.partly_cloudy])
        y_pred = self.tag_store.strs_to_binary(
            [self.dataset.bare_ground, self.dataset.cultivation])

        add_pred_tags, remove_pred_tags = \
            self.tag_store.get_tag_diff(y_true, y_pred)
        self.assertEqual(add_pred_tags, [self.dataset.cultivation])
        self.assertEqual(remove_pred_tags, [self.dataset.partly_cloudy])

    def test_compute_train_probs(self):
        ind3 = 'ind3'
        self.tag_store.add_csv_row((ind3, 'artisinal_mine'))
        rare_sample_prob = 0.5
        sample_probs = \
            self.tag_store.compute_sample_probs(
                [self.ind1, self.ind2, ind3], rare_sample_prob)
        self.assertTrue(np.array_equal(sample_probs, [0.25, 0.25, 0.5]))


if __name__ == '__main__':
    unittest.main()
