import unittest
from os.path import join

import numpy as np

from rastervision.core import Box
from rastervision.core.data import (
    IdentityCRSTransformer, RasterizedSourceConfig, RasterizerConfig,
    GeoJSONVectorSourceConfig, ClassConfig, ClassInferenceTransformerConfig,
    BufferTransformerConfig)
from rastervision.pipeline.file_system import json_to_file
from rastervision.pipeline import rv_config

from tests import data_file_path


class TestRasterizedSourceConfig(unittest.TestCase):
    def test_ensure_required_transformers(self):
        uri = data_file_path('bboxes.geojson')
        cfg = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(uri=uri),
            rasterizer_config=RasterizerConfig(background_class_id=0))
        tfs = cfg.vector_source.transformers
        has_inf_tf = any(
            isinstance(tf, ClassInferenceTransformerConfig) for tf in tfs)
        has_buf_tf = any(isinstance(tf, BufferTransformerConfig) for tf in tfs)
        self.assertTrue(has_inf_tf)
        self.assertTrue(has_buf_tf)


class TestRasterizedSource(unittest.TestCase):
    def setUp(self):
        self.crs_transformer = IdentityCRSTransformer()
        self.extent = Box.make_square(0, 0, 10)
        self.tmp_dir_obj = rv_config.get_tmp_dir()
        self.tmp_dir = self.tmp_dir_obj.name
        self.class_id = 0
        self.background_class_id = 1
        self.line_buffer = 1
        self.class_config = ClassConfig(names=['a'])
        self.uri = join(self.tmp_dir, 'tmp.json')

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def build_source(self, geojson, all_touched=False):
        json_to_file(geojson, self.uri)

        config = RasterizedSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(uri=self.uri),
            rasterizer_config=RasterizerConfig(
                background_class_id=self.background_class_id,
                all_touched=all_touched))
        config.update()
        source = config.build(self.class_config, self.crs_transformer,
                              self.extent)
        return source

    def test_get_chip(self):
        geojson = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[0., 0.], [0., 5.], [5., 5.], [5., 0.],
                                     [0., 0.]]]
                },
                'properties': {
                    'class_id': self.class_id,
                }
            }, {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[7., 0.], [7., 9.]]
                },
                'properties': {
                    'class_id': self.class_id
                }
            }]
        }

        source = self.build_source(geojson)
        self.assertEqual(source.get_extent(), self.extent)
        chip = source.get_image_array()
        self.assertEqual(chip.shape, (10, 10, 1))

        expected_chip = self.background_class_id * np.ones((10, 10, 1))
        expected_chip[0:5, 0:5, 0] = self.class_id
        expected_chip[0:10, 6:8] = self.class_id
        np.testing.assert_array_equal(chip, expected_chip)

    def test_get_chip_no_polygons(self):
        geojson = {'type': 'FeatureCollection', 'features': []}

        source = self.build_source(geojson)
        # Get chip that partially overlaps extent. Expect that chip has zeros
        # outside of extent, and background_class_id otherwise.
        self.assertEqual(source.get_extent(), self.extent)
        chip = source.get_chip(Box.make_square(5, 5, 10))
        self.assertEqual(chip.shape, (10, 10, 1))

        expected_chip = np.full((10, 10, 1), self.background_class_id)
        np.testing.assert_array_equal(chip, expected_chip)

    def test_get_chip_all_touched(self):
        geojson = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[0., 0.], [0., 0.4], [0.4, 0.4],
                                     [0.4, 0.], [0., 0.]]]
                },
                'properties': {
                    'class_id': self.class_id,
                }
            }]
        }

        false_source = self.build_source(geojson, all_touched=False)
        true_source = self.build_source(geojson, all_touched=True)
        chip = false_source.get_image_array()
        expected_chip = self.background_class_id * np.ones((10, 10, 1))
        np.testing.assert_array_equal(chip, expected_chip)

        chip = true_source.get_image_array()
        expected_chip = self.background_class_id * np.ones((10, 10, 1))
        expected_chip[0:1, 0:1, 0] = self.class_id
        np.testing.assert_array_equal(chip, expected_chip)


if __name__ == '__main__':
    unittest.main()
