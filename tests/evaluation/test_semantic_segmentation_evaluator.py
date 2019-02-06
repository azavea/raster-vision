import unittest
from os.path import join
import json

import numpy as np

from rastervision.core.class_map import (ClassItem, ClassMap)
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from rastervision.data import Scene
from tests.mock import (MockRasterSource, MockRasterizedSource)
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

    def get_vector_scene(self, class_id):
        gt_uri = data_file_path('{}-gt-polygons.geojson'.format(class_id))
        pred_uri = data_file_path('{}-pred-polygons.geojson'.format(class_id))

        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 3], num_channels=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        gt_rs = MockRasterizedSource(gt_uri)
        gt_ls = SemanticSegmentationLabelSource(source=gt_rs)

        pred_rs = MockRasterizedSource(pred_uri)
        pred_ls = SemanticSegmentationLabelSource(source=pred_rs)
        pred_ls.vector_output = [{
            'uri': pred_uri,
            'denoise': 0,
            'mode': 'polygons',
            'class_id': class_id
        }]

        return Scene(scene_id, rs, gt_ls, pred_ls)

    def test_evaluator(self):
        class_map = ClassMap([
            ClassItem(id=1, name='one'),
            ClassItem(id=2, name='two'),
        ])
        output_uri = join(self.tmp_dir.name, 'out.json')
        scenes = [self.get_scene(1), self.get_scene(2)]
        evaluator = SemanticSegmentationEvaluator(class_map, output_uri, None)
        evaluator.process(scenes, self.tmp_dir.name)
        eval_json = json.loads(file_to_str(output_uri))
        exp_eval_json = json.loads(
            file_to_str(data_file_path('expected-eval.json')))
        self.assertDictEqual(eval_json, exp_eval_json)

    def test_vector_evaluator(self):
        class_map = ClassMap([
            ClassItem(id=1, name='one'),
            ClassItem(id=2, name='two'),
        ])
        output_uri = join(self.tmp_dir.name, 'raster-out.json')
        vector_output_uri = join(self.tmp_dir.name, 'vector-out.json')
        scenes = [self.get_vector_scene(1), self.get_vector_scene(2)]
        evaluator = SemanticSegmentationEvaluator(class_map, output_uri,
                                                  vector_output_uri)
        evaluator.process(scenes, self.tmp_dir.name)
        vector_eval_json = json.loads(file_to_str(vector_output_uri))
        exp_vector_eval_json = json.loads(
            file_to_str(data_file_path('expected-vector-eval.json')))
        # NOTE:  The precision  and recall  values found  in the  file
        # `expected-vector-eval.json`  are equal to fractions of  the
        # form (n-1)/n for  n <= 7 which  can be seen to  be (and have
        # been manually verified to be) correct.
        self.assertDictEqual(vector_eval_json, exp_vector_eval_json)


if __name__ == '__main__':
    unittest.main()
