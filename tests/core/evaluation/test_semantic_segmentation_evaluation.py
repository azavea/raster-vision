import unittest

import numpy as np

from rastervision.core.data import (ClassConfig,
                                    SemanticSegmentationLabelSource)
from rastervision.core.evaluation import SemanticSegmentationEvaluation
from tests.core.data.mock_raster_source import MockRasterSource


class TestSemanticSegmentationEvaluation(unittest.TestCase):
    def test_compute(self):
        class_config = ClassConfig(names=['one', 'two'])
        class_config.update()
        class_config.ensure_null_class()

        gt_array = np.zeros((4, 4, 1), dtype=np.uint8)
        gt_array[2, 2, 0] = 1
        gt_array[0, 0, 0] = 2
        gt_raster = MockRasterSource([0], 1)
        gt_raster.set_raster(gt_array)
        gt_label_source = SemanticSegmentationLabelSource(
            gt_raster, class_config)

        p_array = np.zeros((4, 4, 1), dtype=np.uint8)
        p_array[1, 1, 0] = 1
        p_raster = MockRasterSource([0], 1)
        p_raster.set_raster(p_array)
        p_label_source = SemanticSegmentationLabelSource(
            p_raster, class_config)

        eval = SemanticSegmentationEvaluation(class_config)
        eval.compute(gt_label_source.get_labels(), p_label_source.get_labels())

        tp0 = 16 - 3  # 4*4 - 3 true positives for class 0
        fp0 = 2  # 1 false positive (2,2) and one at (0,0)
        fn0 = 1  # one false negative (1,1)
        precision0 = float(tp0) / (tp0 + fp0)
        recall0 = float(tp0) / (tp0 + fn0)
        f10 = 2 * float(precision0 * recall0) / (precision0 + recall0)

        eval_item0 = eval.class_to_eval_item[0]
        self.assertAlmostEqual(precision0, eval_item0.precision)
        self.assertAlmostEqual(recall0, eval_item0.recall)
        self.assertAlmostEqual(f10, eval_item0.f1)

        tp1 = 0  # 0 true positives for class 1
        fn1 = 1  # one false negative (2,2)
        precision1 = 0  # float(tp1) / (tp1 + fp1) where fp1 == 1
        recall1 = float(tp1) / (tp1 + fn1)

        eval_item1 = eval.class_to_eval_item[1]
        self.assertEqual(precision1, eval_item1.precision)
        self.assertAlmostEqual(recall1, eval_item1.recall)
        self.assertTrue(np.isnan(eval_item1.f1))

        recall2 = 0
        eval_item2 = eval.class_to_eval_item[2]
        self.assertEqual(recall2, eval_item2.recall)
        self.assertTrue(np.isnan(eval_item2.precision))
        self.assertTrue(np.isnan(eval_item2.f1))

        avg_conf_mat = np.array([[13., 1, 0], [1, 0, 0], [1, 0, 0]])
        avg_recall = (
            (14 / 16) * recall0 + (1 / 16) * recall1 + (1 / 16) * recall2)
        np.testing.assert_array_equal(avg_conf_mat,
                                      np.array(eval.avg_item['conf_mat']))
        self.assertEqual(avg_recall, eval.avg_item['metrics']['recall'])


if __name__ == '__main__':
    unittest.main()
