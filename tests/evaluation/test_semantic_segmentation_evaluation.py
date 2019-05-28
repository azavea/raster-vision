import unittest

import numpy as np

from rastervision.core.class_map import (ClassItem, ClassMap)
from rastervision.evaluation.semantic_segmentation_evaluation import (
    SemanticSegmentationEvaluation)
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from tests.mock import MockRasterSource
from tests import data_file_path


class TestSemanticSegmentationEvaluation(unittest.TestCase):
    def test_compute(self):
        class_map = ClassMap(
            [ClassItem(id=1, name='one'),
             ClassItem(id=2, name='two')])

        # Mismatches: 0 -> 1, 2 -> 1, 1 -> 0
        gt_array = np.ones((4, 4, 1), dtype=np.uint8)
        gt_array[0, 0, 0] = 0
        gt_array[2, 2, 0] = 2
        gt_raster = MockRasterSource([0], 1)
        gt_raster.set_raster(gt_array)
        gt_label_source = SemanticSegmentationLabelSource(source=gt_raster)

        p_array = np.ones((4, 4, 1), dtype=np.uint8)
        p_array[1, 1, 0] = 0
        p_raster = MockRasterSource([0], 1)
        p_raster.set_raster(p_array)
        p_label_source = SemanticSegmentationLabelSource(source=p_raster)

        eval = SemanticSegmentationEvaluation(class_map)
        eval.compute(gt_label_source.get_labels(), p_label_source.get_labels())

        tp1 = 16 - 3  # 4*4 - 3 true positives for class 1
        fp1 = 1  # 1 false positive (2,2) and one don't care at (0,0)
        fn1 = 1  # one false negative (1,1)
        precision1 = float(tp1) / (tp1 + fp1)
        recall1 = float(tp1) / (tp1 + fn1)
        f11 = 2 * float(precision1 * recall1) / (precision1 + recall1)

        tp2 = 0  # 0 true positives for class 2
        fn2 = 1  # one false negative (2,2)
        precision2 = None  # float(tp2) / (tp2 + fp2) where fp2 == 0
        recall2 = float(tp2) / (tp2 + fn2)
        f12 = None

        self.assertAlmostEqual(precision1,
                               eval.class_to_eval_item[1].precision)
        self.assertAlmostEqual(recall1, eval.class_to_eval_item[1].recall)
        self.assertAlmostEqual(f11, eval.class_to_eval_item[1].f1)

        self.assertEqual(precision2, eval.class_to_eval_item[2].precision)
        self.assertAlmostEqual(recall2, eval.class_to_eval_item[2].recall)
        self.assertAlmostEqual(f12, eval.class_to_eval_item[2].f1)

        avg_conf_mat = np.array([[1., 13, 0], [0, 1, 0]])
        avg_recall = (14 / 15) * recall1 + (1 / 15) * recall2
        self.assertTrue(np.array_equal(avg_conf_mat, eval.avg_item.conf_mat))
        self.assertEqual(avg_recall, eval.avg_item.recall)

    def test_compute_ignore_class(self):
        # All ones except for a zero
        gt_array = np.ones((4, 4, 1), dtype=np.uint8)
        gt_array[0, 0, 0] = 0
        gt_raster = MockRasterSource([0], 1)
        gt_raster.set_raster(gt_array)
        gt_label_source = SemanticSegmentationLabelSource(source=gt_raster)

        # All ones
        pred_array = np.ones((4, 4, 1), dtype=np.uint8)
        pred_raster = MockRasterSource([0], 1)
        pred_raster.set_raster(pred_array)
        pred_label_source = SemanticSegmentationLabelSource(source=pred_raster)

        class_map = ClassMap(
            [ClassItem(id=0, name='ignore'),
             ClassItem(id=1, name='one')])
        eval = SemanticSegmentationEvaluation(class_map)
        eval.compute(gt_label_source.get_labels(),
                     pred_label_source.get_labels())
        self.assertAlmostEqual(1, len(eval.class_to_eval_item))
        self.assertAlmostEqual(1.0, eval.class_to_eval_item[1].f1)
        self.assertAlmostEqual(1.0, eval.avg_item.f1)

    def test_vector_compute(self):
        class_map = ClassMap([ClassItem(id=1, name='one', color='#000021')])
        gt_uri = data_file_path('3-gt-polygons.geojson')
        pred_uri = data_file_path('3-pred-polygons.geojson')

        eval = SemanticSegmentationEvaluation(class_map)
        eval.compute_vector(gt_uri, pred_uri, 'polygons', 1)

        # NOTE: The  two geojson files referenced  above contain three
        # unique geometries total, each  file contains two geometries,
        # and there is one geometry shared between the two.
        tp = 1.0
        fp = 1.0
        fn = 1.0
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)

        self.assertAlmostEqual(precision, eval.class_to_eval_item[1].precision)
        self.assertAlmostEqual(recall, eval.class_to_eval_item[1].recall)


if __name__ == '__main__':
    unittest.main()
