import unittest
from os.path import join
import json

import numpy as np

from rastervision.core.class_map import (ClassItem, ClassMap)
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from rastervision.data import Scene
from tests.mock import MockRasterSource
from rastervision.evaluation import SemanticSegmentationEvaluator
from rastervision.rv_config import RVConfig
from rastervision.utils.files import file_to_str
from tests import data_file_path


class TestSemanticSegmentationEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = RVConfig.get_tmp_dir()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def get_scene(self, class_id):
        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 3], num_channels=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        gt_rs = MockRasterSource(channel_order=[0], num_channels=1)
        gt_arr = np.full((10, 10, 1), class_id)
        gt_rs.set_raster(gt_arr)
        gt_ls = SemanticSegmentationLabelSource(source=gt_rs)

        pred_rs = MockRasterSource(channel_order=[0], num_channels=1)
        pred_arr = np.ones((10, 10, 1))
        pred_arr[5:10, :, :] = 2
        pred_rs.set_raster(pred_arr)
        pred_ls = SemanticSegmentationLabelSource(source=pred_rs)

        return Scene(scene_id, rs, gt_ls, pred_ls)

    def test_evaluator(self):
        class_map = ClassMap([
            ClassItem(id=1, name='one'),
            ClassItem(id=2, name='two'),
        ])
        output_uri = join(self.tmp_dir.name, 'out.json')
        scenes = [self.get_scene(1), self.get_scene(2)]
        evaluator = SemanticSegmentationEvaluator(class_map, output_uri)
        evaluator.process(scenes, self.tmp_dir.name)
        eval_json = json.loads(file_to_str(output_uri))
        exp_eval_json = json.loads(
            file_to_str(data_file_path('expected-eval.json')))
        self.assertDictEqual(eval_json, exp_eval_json)


if __name__ == '__main__':
    unittest.main()
