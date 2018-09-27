import unittest

import numpy as np

from rastervision.evaluation import ObjectDetectionEvaluation
from rastervision.core.class_map import ClassItem, ClassMap
from rastervision.core.box import Box
from rastervision.data.label import ObjectDetectionLabels


class TestObjectDetectionEvaluation(unittest.TestCase):
    def make_class_map(self):
        class_items = [ClassItem(1, 'car'), ClassItem(2, 'building')]
        return ClassMap(class_items)

    def make_ground_truth_labels(self):
        size = 100
        nw = Box.make_square(0, 0, size)
        ne = Box.make_square(0, 200, size)
        se = Box.make_square(200, 200, size)
        sw = Box.make_square(200, 0, size)
        npboxes = Box.to_npboxes([nw, ne, se, sw])
        class_ids = np.array([1, 1, 2, 2])
        return ObjectDetectionLabels(npboxes, class_ids)

    def make_predicted_labels(self):
        size = 100
        # Predicted labels are only there for three of the ground truth boxes,
        # and are offset by 10 pixels.
        nw = Box.make_square(10, 0, size)
        ne = Box.make_square(10, 200, size)
        se = Box.make_square(210, 200, size)
        npboxes = Box.to_npboxes([nw, ne, se])
        class_ids = np.array([1, 1, 2])
        scores = np.ones(class_ids.shape)
        return ObjectDetectionLabels(npboxes, class_ids, scores=scores)

    def test_compute(self):
        class_map = self.make_class_map()
        eval = ObjectDetectionEvaluation(class_map)
        gt_labels = self.make_ground_truth_labels()
        pred_labels = self.make_predicted_labels()

        eval.compute(gt_labels, pred_labels)
        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(eval_item1.gt_count, 2)
        self.assertEqual(eval_item1.precision, 1.0)
        self.assertEqual(eval_item1.recall, 1.0)
        self.assertEqual(eval_item1.f1, 1.0)

        eval_item2 = eval.class_to_eval_item[2]
        self.assertEqual(eval_item2.gt_count, 2)
        self.assertEqual(eval_item2.precision, 1.0)
        self.assertEqual(eval_item2.recall, 0.5)
        self.assertEqual(eval_item2.f1, 2 / 3)

        avg_item = eval.avg_item
        self.assertEqual(avg_item.gt_count, 4)
        self.assertAlmostEqual(avg_item.precision, 1.0)
        self.assertEqual(avg_item.recall, 0.75)
        self.assertAlmostEqual(avg_item.f1, 0.83, places=2)

    def test_compute_no_preds(self):
        class_map = self.make_class_map()
        eval = ObjectDetectionEvaluation(class_map)
        gt_labels = self.make_ground_truth_labels()
        pred_labels = ObjectDetectionLabels.make_empty()

        eval.compute(gt_labels, pred_labels)
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
        self.assertEqual(avg_item.precision, 0.0)
        self.assertEqual(avg_item.recall, 0.0)
        self.assertEqual(avg_item.f1, 0.0)

    def test_compute_no_ground_truth(self):
        class_map = self.make_class_map()
        eval = ObjectDetectionEvaluation(class_map)
        gt_labels = ObjectDetectionLabels.make_empty()
        pred_labels = self.make_predicted_labels()

        eval.compute(gt_labels, pred_labels)
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


if __name__ == '__main__':
    unittest.main()
