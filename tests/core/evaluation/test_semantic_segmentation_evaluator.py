import unittest
from os.path import join

import numpy as np
from shapely.geometry import shape

from rastervision.pipeline.file_system import file_to_json, get_tmp_dir
from rastervision.core.data import ClassConfig
from rastervision.core import Box
from rastervision.core.data import (
    Scene, IdentityCRSTransformer, SemanticSegmentationLabelSource,
    RasterizedSourceConfig, RasterizerConfig, GeoJSONVectorSourceConfig,
    PolygonVectorOutputConfig, ClassInferenceTransformerConfig)
from rastervision.core.evaluation import SemanticSegmentationEvaluator

from tests.core.data.mock_raster_source import (MockRasterSource)
from tests import data_file_path


class MockRVPipelineConfig:
    eval_uri = '/abc/def/eval'


class TestSemanticSegmentationEvaluator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = get_tmp_dir()

        self.class_config = ClassConfig(names=['one', 'two'])
        self.class_config.update()
        self.class_config.ensure_null_class()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def get_scene(self, class_id):
        # Make scene where ground truth is all set to class_id
        # and predictions are set to half 0's and half 1's
        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 2], num_channels_raw=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        gt_rs = MockRasterSource(channel_order=[0], num_channels_raw=1)
        gt_arr = np.full((10, 10, 1), class_id)
        gt_rs.set_raster(gt_arr)
        gt_ls = SemanticSegmentationLabelSource(gt_rs, self.class_config)

        pred_rs = MockRasterSource(channel_order=[0], num_channels_raw=1)
        pred_arr = np.zeros((10, 10, 1))
        pred_arr[5:10, :, :] = 1
        pred_rs.set_raster(pred_arr)
        pred_ls = SemanticSegmentationLabelSource(pred_rs, self.class_config)

        return Scene(scene_id, rs, gt_ls, pred_ls)

    def test_evaluator(self):
        output_uri = join(self.tmp_dir.name, 'out.json')
        scenes = [self.get_scene(0), self.get_scene(1)]

        # the mock scene returned by get_scene uses a label source in place of
        # a label store, but an SS label store is expected to have a
        # vector_outputs, so we add that here
        scenes[0].label_store.vector_outputs = None
        scenes[1].label_store.vector_outputs = None

        evaluator = SemanticSegmentationEvaluator(self.class_config,
                                                  output_uri)
        evaluator.process(scenes, self.tmp_dir.name)
        eval_json = file_to_json(output_uri)
        exp_eval_json = file_to_json(data_file_path('expected-eval.json'))
        self.assertDictEqual(eval_json, exp_eval_json)

    def get_vector_scene(self, class_id, use_aoi=False):
        gt_uri = data_file_path('{}-gt-polygons.geojson'.format(class_id))
        pred_uri = data_file_path('{}-pred-polygons.geojson'.format(class_id))

        scene_id = str(class_id)
        rs = MockRasterSource(channel_order=[0, 1, 2], num_channels_raw=3)
        rs.set_raster(np.zeros((10, 10, 3)))

        crs_transformer = IdentityCRSTransformer()
        extent = Box.make_square(0, 0, 360)

        config = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uris=gt_uri,
                transformers=[
                    ClassInferenceTransformerConfig(default_class_id=0)
                ]),
            rasterizer_config=RasterizerConfig(background_class_id=1))
        gt_rs = config.build(self.class_config, crs_transformer, extent)
        gt_ls = SemanticSegmentationLabelSource(gt_rs, self.class_config)

        config = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uris=pred_uri,
                transformers=[
                    ClassInferenceTransformerConfig(default_class_id=0)
                ]),
            rasterizer_config=RasterizerConfig(background_class_id=1))
        pred_rs = config.build(self.class_config, crs_transformer, extent)
        pred_ls = SemanticSegmentationLabelSource(pred_rs, self.class_config)
        pred_ls.vector_outputs = [
            PolygonVectorOutputConfig(
                uri=pred_uri, denoise=0, class_id=class_id)
        ]

        if use_aoi:
            aoi_uri = data_file_path('{}-aoi.geojson'.format(class_id))
            aoi_geojson = file_to_json(aoi_uri)
            aoi_polygons = [shape(aoi_geojson['features'][0]['geometry'])]
            return Scene(scene_id, rs, gt_ls, pred_ls, aoi_polygons)

        return Scene(scene_id, rs, gt_ls, pred_ls)


if __name__ == '__main__':
    unittest.main()
