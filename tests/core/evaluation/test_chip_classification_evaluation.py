import unittest

from rastervision.core.evaluation import ChipClassificationEvaluation
from rastervision.core.data.class_config import ClassConfig
from rastervision.core import Box
from rastervision.core.data.label import ChipClassificationLabels


class TestChipClassificationEvaluation(unittest.TestCase):
    def make_class_config(self):
        return ClassConfig(names=['grassy', 'urban'])

    def make_labels(self, class_ids):
        """Make 2x2 grid label store.

        Args:
            class_ids: 2x2 array of class_ids to use
        """
        cell_size = 200
        y_cells = 2
        x_cells = 2
        labels = ChipClassificationLabels()

        for yind in range(y_cells):
            for xind in range(x_cells):
                ymin = yind * cell_size
                xmin = xind * cell_size
                ymax = ymin + cell_size
                xmax = xmin + cell_size
                window = Box(ymin, xmin, ymax, xmax)
                class_id = class_ids[yind][xind]
                new_labels = ChipClassificationLabels()
                new_labels.set_cell(window, class_id)
                labels.extend(new_labels)

        return labels

    def assert_eval_single_null(self, eval):
        eval_item0 = eval.class_to_eval_item[0]
        self.assertEqual(eval_item0.gt_count, 2)
        self.assertEqual(eval_item0.precision, 1.0)
        self.assertEqual(eval_item0.recall, 0.5)
        self.assertAlmostEqual(eval_item0.f1, 2 / 3, places=2)

        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(eval_item1.gt_count, 1)
        self.assertEqual(eval_item1.precision, 0.5)
        self.assertEqual(eval_item1.recall, 1.0)
        self.assertAlmostEqual(eval_item1.f1, 2 / 3, places=2)

        avg_item = eval.avg_item
        self.assertEqual(avg_item.gt_count, 3)
        self.assertAlmostEqual(avg_item.precision, 0.83, places=2)
        self.assertAlmostEqual(avg_item.recall, 2 / 3, places=2)
        self.assertAlmostEqual(avg_item.f1, 2 / 3, places=2)

    def test_compute_single_pred_null(self):
        class_config = self.make_class_config()
        eval = ChipClassificationEvaluation(class_config)
        gt_class_ids = [[0, 1], [0, 1]]
        gt_labels = self.make_labels(gt_class_ids)
        pred_class_ids = [[0, None], [1, 1]]
        pred_labels = self.make_labels(pred_class_ids)
        eval.compute(gt_labels, pred_labels)
        self.assert_eval_single_null(eval)

    def test_compute_single_gt_null(self):
        class_config = self.make_class_config()
        eval = ChipClassificationEvaluation(class_config)
        gt_class_ids = [[0, None], [0, 1]]
        gt_labels = self.make_labels(gt_class_ids)
        pred_class_ids = [[0, 1], [1, 1]]
        pred_labels = self.make_labels(pred_class_ids)
        eval.compute(gt_labels, pred_labels)
        self.assert_eval_single_null(eval)


if __name__ == '__main__':
    unittest.main()
