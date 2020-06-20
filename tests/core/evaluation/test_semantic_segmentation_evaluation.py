import unittest

import numpy as np

from rastervision.core.data import ClassConfig
from rastervision.core.evaluation import SemanticSegmentationEvaluation
from rastervision.core.data import SemanticSegmentationLabelSource
from tests.core.data.mock_raster_source import MockRasterSource
from tests import data_file_path


class TestSemanticSegmentationEvaluation(unittest.TestCase):
    def test_compute(self):
        class_config = ClassConfig(names=['one', 'two'])
        class_config.update()
        class_config.ensure_null_class()
        null_class_id = class_config.get_null_class_id()

        gt_array = np.zeros((4, 4, 1), dtype=np.uint8)
        gt_array[2, 2, 0] = 1
        gt_array[0, 0, 0] = 2
        gt_raster = MockRasterSource([0], 1)
        gt_raster.set_raster(gt_array)
        gt_label_source = SemanticSegmentationLabelSource(
            gt_raster, null_class_id)

        p_array = np.zeros((4, 4, 1), dtype=np.uint8)
        p_array[1, 1, 0] = 1
        p_raster = MockRasterSource([0], 1)
        p_raster.set_raster(p_array)
        p_label_source = SemanticSegmentationLabelSource(
            p_raster, null_class_id)

        eval = SemanticSegmentationEvaluation(class_config)
        eval.compute(gt_label_source.get_labels(), p_label_source.get_labels())

        tp0 = 16 - 3  # 4*4 - 3 true positives for class 0
        fp0 = 1  # 1 false positive (2,2) and one don't care at (0,0)
        fn0 = 1  # one false negative (1,1)
        precision0 = float(tp0) / (tp0 + fp0)
        recall0 = float(tp0) / (tp0 + fn0)
        f10 = 2 * float(precision0 * recall0) / (precision0 + recall0)

        tp1 = 0  # 0 true positives for class 1
        fn1 = 1  # one false negative (2,2)
        precision1 = 0  # float(tp1) / (tp1 + fp1) where fp1 == 1
        recall1 = float(tp1) / (tp1 + fn1)
        f11 = None

        self.assertAlmostEqual(precision0,
                               eval.class_to_eval_item[0].precision)
        self.assertAlmostEqual(recall0, eval.class_to_eval_item[0].recall)
        self.assertAlmostEqual(f10, eval.class_to_eval_item[0].f1)

        self.assertEqual(precision1, eval.class_to_eval_item[1].precision)
        self.assertAlmostEqual(recall1, eval.class_to_eval_item[1].recall)
        self.assertAlmostEqual(f11, eval.class_to_eval_item[1].f1)

        avg_conf_mat = np.array([[0, 0, 0], [13., 1, 0], [1, 0, 0]])
        avg_recall = (14 / 15) * recall0 + (1 / 15) * recall1
        self.assertTrue(np.array_equal(avg_conf_mat, eval.avg_item.conf_mat))
        self.assertEqual(avg_recall, eval.avg_item.recall)

    def test_compute_ignore_class(self):
        class_config = ClassConfig(names=['one', 'two'])
        class_config.update()
        class_config.ensure_null_class()
        null_class_id = class_config.get_null_class_id()

        gt_array = np.zeros((4, 4, 1), dtype=np.uint8)
        gt_array[0, 0, 0] = 2
        gt_raster = MockRasterSource([0], 1)
        gt_raster.set_raster(gt_array)
        gt_label_source = SemanticSegmentationLabelSource(
            gt_raster, null_class_id)

        pred_array = np.zeros((4, 4, 1), dtype=np.uint8)
        pred_array[0, 0, 0] = 1
        pred_raster = MockRasterSource([0], 1)
        pred_raster.set_raster(pred_array)
        pred_label_source = SemanticSegmentationLabelSource(
            pred_raster, null_class_id)

        eval = SemanticSegmentationEvaluation(class_config)
        eval.compute(gt_label_source.get_labels(),
                     pred_label_source.get_labels())
        self.assertAlmostEqual(1.0, eval.class_to_eval_item[0].f1)
        self.assertAlmostEqual(1.0, eval.avg_item.f1)

    def test_vector_compute(self):
        class_config = ClassConfig(names=['one', 'two'])
        class_config.update()
        class_config.ensure_null_class()

        gt_uri = data_file_path('2-gt-polygons.geojson')
        pred_uri = data_file_path('2-pred-polygons.geojson')

        eval = SemanticSegmentationEvaluation(class_config)
        eval.compute_vector(gt_uri, pred_uri, 'polygons', 0)

        # NOTE: The  two geojson files referenced  above contain three
        # unique geometries total, each  file contains two geometries,
        # and there is one geometry shared between the two.
        tp = 1.0
        fp = 1.0
        fn = 1.0
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)

        self.assertAlmostEqual(precision, eval.class_to_eval_item[0].precision)
        self.assertAlmostEqual(recall, eval.class_to_eval_item[0].recall)


if __name__ == '__main__':
    unittest.main()
