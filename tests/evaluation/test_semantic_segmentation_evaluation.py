import unittest

import numpy as np

from rastervision.core.class_map import (ClassItem, ClassMap)
from rastervision.evaluation.semantic_segmentation_evaluation import (
    SemanticSegmentationEvaluation)
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from rastervision.data.raster_source.raster_source import RasterSource
from rastervision.core.box import Box

# from ..data.label_source.test_semantic_segmentation_label_source import (
#    MockRasterSource)


# This class was copied from test_semantic_segmentation_label_source
# because it can't be imported because
# SystemError: Parent module '' not loaded, cannot perform relative import
# TODO: solve this problem
class MockRasterSource(RasterSource):
    def __init__(self, zeros=False, data=None):
        if data is not None:
            self.data = data
            (self.height, self.width, self.channels) = data.shape
        elif zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.zeros(
                (self.height, self.width, self.channels), dtype=np.uint8)
        elif not zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.random.randint(
                0,
                2,
                size=(self.width, self.height, self.channels),
                dtype=np.uint8)
            self.data[:, :, 0:(self.channels - 1)] = np.zeros(
                (self.height, self.width, self.channels - 1), dtype=np.uint8)

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def _get_chip(self, window):
        ymin = window.ymin
        xmin = window.xmin
        ymax = window.ymax
        xmax = window.xmax
        return self.data[ymin:ymax, xmin:xmax, :]

    def get_chip(self, window):
        return self.get_chip(window)

    def get_crs_transformer(self, window):
        return None

    def get_dtype(self):
        return np.uint8


class TestSemanticSegmentationEvaluation(unittest.TestCase):
    def test_compute(self):
        class_map = ClassMap([
            ClassItem(id=1, name='one', color='#010101'),
            ClassItem(id=2, name='two', color='#020202')
        ])

        gt_array = np.ones((4, 4, 3), dtype=np.uint8)
        gt_array[0, 0, :] = 0
        gt_array[2, 2, :] = 2
        gt_raster = MockRasterSource(data=gt_array)
        gt_label_source = SemanticSegmentationLabelSource(
            source=gt_raster, rgb_class_map=class_map)

        p_array = np.ones((4, 4, 3), dtype=np.uint8)
        p_array[1, 1, :] = 0
        p_raster = MockRasterSource(data=p_array)
        p_label_source = SemanticSegmentationLabelSource(
            source=p_raster, rgb_class_map=class_map)

        eval = SemanticSegmentationEvaluation(class_map)
        eval.compute(gt_label_source.get_labels(), p_label_source.get_labels())

        tp1 = 16 - 3  # 4*4 - 3 true positives for class 1
        fp1 = 1  # 1 false positive (2,2) and one don't care at (0,0)
        fn1 = 1  # one false negative (1,1)
        precision1 = float(tp1) / (tp1 + fp1)
        recall1 = float(tp1) / (tp1 + fn1)

        tp2 = 0  # 0 true positives for class 2
        fn2 = 1  # one false negative (2,2)
        precision2 = None  # float(tp2) / (tp2 + fp2) where fp2 == 0
        recall2 = float(tp2) / (tp2 + fn2)

        self.assertAlmostEqual(precision1,
                               eval.class_to_eval_item[1].precision)
        self.assertAlmostEqual(recall1, eval.class_to_eval_item[1].recall)
        self.assertEqual(precision2, eval.class_to_eval_item[2].precision)
        self.assertAlmostEqual(recall2, eval.class_to_eval_item[2].recall)


if __name__ == '__main__':
    unittest.main()
