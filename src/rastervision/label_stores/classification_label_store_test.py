import unittest

import numpy as np

from rastervision.label_stores.classification_label_store import (
    ClassificationLabelStore)
from rastervision.labels.classification_labels import ClassificationLabels
from rastervision.core.box import Box


class TestClassificationLabelStore(unittest.TestCase):
    def setUp(self):
        self.box1 = Box.make_square(0, 0, 2)
        self.box2 = Box.make_square(2, 2, 2)
        self.class_id1 = 1
        self.class_id2 = 2

        self.labels = ClassificationLabels()
        self.labels.set_cell(self.box1, self.class_id1)
        self.labels.set_cell(self.box2, self.class_id2)

    def tearDown(self):
        pass

    def test_constructor(self):
        label_store = ClassificationLabelStore()
        self.assertEqual(len(label_store.get_labels()), 0)

    def test_set_get_clear(self):
        label_store = ClassificationLabelStore()

        label_store.set_labels(self.labels)
        labels = label_store.get_labels()
        self.assertEqual(labels, self.labels)

        label_store.clear()
        self.assertEqual(len(label_store.get_labels()), 0)

    def test_get_window(self):
        label_store = ClassificationLabelStore()

        label_store.set_labels(self.labels)
        labels = label_store.get_labels(window=self.box1)
        self.assertEqual(len(labels), 1)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)

    def test_extend(self):
        label_store = ClassificationLabelStore()

        labels = ClassificationLabels()
        labels.set_cell(self.box1, self.class_id1)
        label_store.extend(labels)

        labels = ClassificationLabels()
        labels.set_cell(self.box2, self.class_id2)
        label_store.extend(labels)

        labels = label_store.get_labels()
        self.assertEqual(len(labels), 2)
        class_id = labels.get_cell_class_id(self.box1)
        self.assertEqual(class_id, self.class_id1)
        class_id = labels.get_cell_class_id(self.box2)
        self.assertEqual(class_id, self.class_id2)


if __name__ == '__main__':
    unittest.main()
