import unittest

import numpy as np

from rastervision.core.class_map import (ClassItem, ClassMap)
from rastervision.evaluations.segmentation_evaluation import (
    SegmentationEvaluation)
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationRasterFile)
from rastervision.label_stores.segmentation_raster_file_test import (
    TestingRasterSource)


class TestSegmentationEvaluation(unittest.TestCase):
    def test_compute(self):
        class_map = ClassMap(
            [ClassItem(id=1, name='one'),
             ClassItem(id=2, name='two')])

        raster_class_map = {'#010101': 1, '#020202': 2}

        gt_array = np.ones((5, 5, 3), dtype=np.uint8)
        gt_array[0, 0, :] = 0
        gt_array[2, 2, :] = 2
        gt_raster = TestingRasterSource(data=gt_array)
        gt_label_store = SegmentationRasterFile(
            source=gt_raster,
            sink=None,
            class_map=class_map,
            raster_class_map=raster_class_map)

        p_array = np.ones((4, 4, 3), dtype=np.uint8)
        p_array[1, 1, :] = 0
        p_raster = TestingRasterSource(data=p_array)
        p_label_store = SegmentationRasterFile(
            source=p_raster,
            sink=None,
            class_map=class_map,
            raster_class_map=raster_class_map)

        seval = SegmentationEvaluation()
        seval.compute(class_map, gt_label_store, p_label_store)

        tp1 = 16 - 3  # 4*4 - 3 true positives for class 1
        fp1 = 2  # 2 false positives (0,0) and (2,2)
        fn1 = 1  # one false negative (1,1)
        precision1 = float(tp1) / (tp1 + fp1)
        recall1 = float(tp1) / (tp1 + fn1)

        tp2 = 0  # 0 true positives for class 2
        fn2 = 1  # one false negative (2,2)
        precision2 = None  # float(tp2) / (tp2 + fp2) where fp2 == 0
        recall2 = float(tp2) / (tp2 + fn2)

        self.assertAlmostEqual(precision1,
                               seval.class_to_eval_item[1].precision)
        self.assertAlmostEqual(recall1, seval.class_to_eval_item[1].recall)
        self.assertEqual(precision2, seval.class_to_eval_item[2].precision)
        self.assertAlmostEqual(recall2, seval.class_to_eval_item[2].recall)


if __name__ == "__main__":
    unittest.main()
