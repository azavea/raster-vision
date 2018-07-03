import unittest

import numpy as np

from rastervision.evaluations.object_detection_evaluation import (
    ObjectDetectionEvaluation)
from rastervision.core.class_map import ClassItem, ClassMap
from rastervision.core.box import Box
from rastervision.label_stores.object_detection_label_store import (
    ObjectDetectionLabelStore)
from rastervision.labels.object_detection_labels import (
    ObjectDetectionLabels)


class TestObjectDetectionEvaluation(unittest.TestCase):
    def make_class_map(self):
        class_items = [
            ClassItem(1, 'car'),
            ClassItem(2, 'building')
        ]
        return ClassMap(class_items)

    def make_ground_truth_label_store(self):
        size = 100
        nw = Box.make_square(0, 0, size)
        ne = Box.make_square(0, 200, size)
        se = Box.make_square(200, 200, size)
        sw = Box.make_square(200, 0, size)
        npboxes = Box.to_npboxes([nw, ne, se, sw])
        class_ids = np.array([1, 1, 2, 2])
        label_store = ObjectDetectionLabelStore()
        window = Box.make_square(0, 0, 1000)
        label_store.extend(window, ObjectDetectionLabels(npboxes, class_ids))
        return label_store

    def make_predicted_label_store(self):
        size = 100
        # Predicted labels are only there for three of the ground truth boxes,
        # and are offset by 10 pixels.
        nw = Box.make_square(10, 0, size)
        ne = Box.make_square(10, 200, size)
        se = Box.make_square(210, 200, size)
        npboxes = Box.to_npboxes([nw, ne, se])
        class_ids = np.array([1, 1, 2])
        scores = np.ones(class_ids.shape)
        label_store = ObjectDetectionLabelStore()
        window = Box.make_square(0, 0, 1000)
        label_store.extend(window, ObjectDetectionLabels(
            npboxes, class_ids, scores=scores))
        return label_store

    def test_compute(self):
        eval = ObjectDetectionEvaluation()
        class_map = self.make_class_map()
        gt_label_store = self.make_ground_truth_label_store()
        pred_label_store = self.make_predicted_label_store()

        eval.compute(
            class_map, gt_label_store, pred_label_store)
        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(eval_item1.gt_count, 2)
        self.assertEqual(eval_item1.precision, 1.0)
        self.assertEqual(eval_item1.recall, 1.0)
        self.assertEqual(eval_item1.f1, 1.0)

        eval_item2 = eval.class_to_eval_item[2]
        self.assertEqual(eval_item2.gt_count, 2)
        self.assertEqual(eval_item2.precision, 1.0)
        self.assertEqual(eval_item2.recall, 0.5)
        self.assertEqual(eval_item2.f1, 2/3)

        avg_item = eval.avg_item
        self.assertEqual(avg_item.gt_count, 4)
        self.assertAlmostEqual(avg_item.precision, 1.0)
        self.assertEqual(avg_item.recall, 0.75)
        self.assertAlmostEqual(avg_item.f1, 0.83, places=2)

    def test_compute_no_preds(self):
        eval = ObjectDetectionEvaluation()
        class_map = self.make_class_map()
        gt_label_store = self.make_ground_truth_label_store()
        pred_label_store = ObjectDetectionLabelStore()

        eval.compute(
            class_map, gt_label_store, pred_label_store)
        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(eval_item1.gt_count, 2)
        self.assertEqual(eval_item1.precision, None)
        self.assertEqual(eval_item1.recall, 0.0)
        self.assertEqual(eval_item1.f1, None)

        eval_item2 = eval.class_to_eval_item[2]
        self.assertEqual(eval_item2.gt_count, 2)
        self.assertEqual(eval_item2.precision, None)
        self.assertEqual(eval_item2.recall, 0.0)
        self.assertEqual(eval_item2.f1, None)

        avg_item = eval.avg_item
        self.assertEqual(avg_item.gt_count, 4)
        self.assertEqual(avg_item.precision, None)
        self.assertEqual(avg_item.recall, 0.0)
        self.assertEqual(avg_item.f1, None)

    def test_compute_no_ground_truth(self):
        eval = ObjectDetectionEvaluation()
        class_map = self.make_class_map()
        gt_label_store = ObjectDetectionLabelStore()
        pred_label_store = self.make_predicted_label_store()

        eval.compute(
            class_map, gt_label_store, pred_label_store)
        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(eval_item1.gt_count, 0)
        self.assertEqual(eval_item1.precision, None)
        self.assertEqual(eval_item1.recall, None)
        self.assertEqual(eval_item1.f1, None)

        eval_item2 = eval.class_to_eval_item[2]
        self.assertEqual(eval_item2.gt_count, 0)
        self.assertEqual(eval_item2.precision, None)
        self.assertEqual(eval_item2.recall, None)
        self.assertEqual(eval_item2.f1, None)

        avg_item = eval.avg_item
        self.assertEqual(avg_item.gt_count, 0)
        self.assertEqual(avg_item.precision, None)
        self.assertEqual(avg_item.recall, None)
        self.assertEqual(avg_item.f1, None)
