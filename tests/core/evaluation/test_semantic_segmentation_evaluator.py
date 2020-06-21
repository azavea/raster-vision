import unittest
from os.path import join

import numpy as np
from shapely.geometry import shape

from rastervision.core.data import ClassConfig
from rastervision.core import Box
from rastervision.core.data import (
    Scene, IdentityCRSTransformer, SemanticSegmentationLabelSource,
    RasterizedSourceConfig, RasterizerConfig, GeoJSONVectorSourceConfig,
    PolygonVectorOutputConfig)
from rastervision.core.evaluation import SemanticSegmentationEvaluator
from rastervision.pipeline import rv_config
from rastervision.pipeline.file_system import file_to_json

from tests.core.data.mock_raster_source import (MockRasterSource)
from tests import data_file_path


class TestSemanticSegmentationEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = rv_config.get_tmp_dir()

        self.class_config = ClassConfig(names=['one', 'two'])
        self.class_config.update()
        self.class_config.ensure_null_class()
        self.null_class_id = self.class_config.get_null_class_id()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def get_scene(self, class_id):
        # Make scene where ground truth is all set to class_id
        # and predictions are set to half 0's and half 1's
        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 2], num_channels=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        gt_rs = MockRasterSource(channel_order=[0], num_channels=1)
        gt_arr = np.full((10, 10, 1), class_id)
        gt_rs.set_raster(gt_arr)
        gt_ls = SemanticSegmentationLabelSource(gt_rs, self.null_class_id)

        pred_rs = MockRasterSource(channel_order=[0], num_channels=1)
        pred_arr = np.zeros((10, 10, 1))
        pred_arr[5:10, :, :] = 1
        pred_rs.set_raster(pred_arr)
        pred_ls = SemanticSegmentationLabelSource(pred_rs, self.null_class_id)

        return Scene(scene_id, rs, gt_ls, pred_ls)

    def test_evaluator(self):
        output_uri = join(self.tmp_dir.name, 'out.json')
        scenes = [self.get_scene(0), self.get_scene(1)]
        evaluator = SemanticSegmentationEvaluator(self.class_config,
                                                  output_uri, None)
        evaluator.process(scenes, self.tmp_dir.name)
        eval_json = file_to_json(output_uri)
        exp_eval_json = file_to_json(data_file_path('expected-eval.json'))
        self.assertDictEqual(eval_json, exp_eval_json)

    def get_vector_scene(self, class_id, use_aoi=False):
        gt_uri = data_file_path('{}-gt-polygons.geojson'.format(class_id))
        pred_uri = data_file_path('{}-pred-polygons.geojson'.format(class_id))

        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 3], num_channels=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        crs_transformer = IdentityCRSTransformer()
        extent = Box.make_square(0, 0, 360)

        config = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=gt_uri, default_class_id=0),
            rasterizer_config=RasterizerConfig(background_class_id=1))
        gt_rs = config.build(self.class_config, crs_transformer, extent)
        gt_ls = SemanticSegmentationLabelSource(gt_rs, self.null_class_id)

        config = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=pred_uri, default_class_id=0),
            rasterizer_config=RasterizerConfig(background_class_id=1))
        pred_rs = config.build(self.class_config, crs_transformer, extent)
        pred_ls = SemanticSegmentationLabelSource(pred_rs, self.null_class_id)
        pred_ls.vector_output = [
            PolygonVectorOutputConfig(
                uri=pred_uri, denoise=0, class_id=class_id)
        ]

        if use_aoi:
            aoi_uri = data_file_path('{}-aoi.geojson'.format(class_id))
            aoi_geojson = file_to_json(aoi_uri)
            aoi_polygons = [shape(aoi_geojson['features'][0]['geometry'])]
            return Scene(scene_id, rs, gt_ls, pred_ls, aoi_polygons)

        return Scene(scene_id, rs, gt_ls, pred_ls)

    def test_vector_evaluator(self):
        output_uri = join(self.tmp_dir.name, 'raster-out.json')
        vector_output_uri = join(self.tmp_dir.name, 'vector-out.json')
        scenes = [self.get_vector_scene(0), self.get_vector_scene(1)]
        evaluator = SemanticSegmentationEvaluator(
            self.class_config, output_uri, vector_output_uri)
        evaluator.process(scenes, self.tmp_dir.name)
        vector_eval_json = file_to_json(vector_output_uri)
        exp_vector_eval_json = file_to_json(
            data_file_path('expected-vector-eval.json'))

        # NOTE:  The precision  and recall  values found  in the  file
        # `expected-vector-eval.json`  are equal to fractions of  the
        # form (n-1)/n for  n <= 7 which  can be seen to  be (and have
        # been manually verified to be) correct.
        self.assertDictEqual(vector_eval_json, exp_vector_eval_json)

    def test_vector_evaluator_with_aoi(self):
        output_uri = join(self.tmp_dir.name, 'raster-out.json')
        vector_output_uri = join(self.tmp_dir.name, 'vector-out.json')
        scenes = [self.get_vector_scene(0, use_aoi=True)]
        evaluator = SemanticSegmentationEvaluator(
            self.class_config, output_uri, vector_output_uri)
        evaluator.process(scenes, self.tmp_dir.name)
        vector_eval_json = file_to_json(vector_output_uri)
        exp_vector_eval_json = file_to_json(
            data_file_path('expected-vector-eval-with-aoi.json'))

        # NOTE:  The precision  and recall  values found  in the  file
        # `expected-vector-eval.json`  are equal to fractions of  the
        # form (n-1)/n for  n <= 7 which  can be seen to  be (and have
        # been manually verified to be) correct.
        self.assertDictEqual(vector_eval_json, exp_vector_eval_json)


if __name__ == '__main__':
    unittest.main()
