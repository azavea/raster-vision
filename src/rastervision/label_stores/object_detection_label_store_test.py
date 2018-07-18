import unittest

import numpy as np

from rastervision.label_stores.object_detection_label_store import (
    ObjectDetectionLabelStore)
from rastervision.labels.object_detection_labels import ObjectDetectionLabels
from rastervision.core.box import Box


class TestObjectDetectionLabelStore(unittest.TestCase):
    def setUp(self):
        self.npboxes = np.array([
            [0., 0., 2., 2.],
            [2., 2., 4., 4.]
        ])
        self.class_ids = np.array([1, 2])
        self.scores = np.array([0.99, 0.9])
        self.labels = ObjectDetectionLabels(
            self.npboxes, self.class_ids, scores=self.scores)

    def test_constructor(self):
        store = ObjectDetectionLabelStore()
        self.assertEqual(len(store.get_labels()), 0)

    def test_extend(self):
        store = ObjectDetectionLabelStore()
        store.extend(self.labels)
        labels = store.get_labels()
        labels.assert_equal(self.labels)

    def test_set(self):
        store = ObjectDetectionLabelStore()
        store.set_labels(self.labels)
        store.set_labels(self.labels)
        labels = store.get_labels()
        self.assertEqual(labels, self.labels)

    def test_get_labels(self):
        store = ObjectDetectionLabelStore()
        store.extend(self.labels)
        labels = store.get_labels()
        labels.assert_equal(self.labels)

        window = Box.make_square(2, 2, 2)
        labels = store.get_labels(window=window)
        npboxes = np.array([
            [2., 2., 4., 4.]
        ])
        class_ids = np.array([2])
        scores = np.array([0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)


if __name__ == '__main__':
    unittest.main()
