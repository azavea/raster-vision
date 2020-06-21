import unittest
import os

import numpy as np

from rastervision.core.data import (ObjectDetectionLabelSourceConfig,
                                    GeoJSONVectorSourceConfig,
                                    ObjectDetectionLabels, ClassConfig)
from rastervision.core import Box
from rastervision.pipeline import rv_config
from rastervision.pipeline.file_system import json_to_file

from tests.core.data.mock_crs_transformer import DoubleCRSTransformer


class TestObjectDetectionLabelSource(unittest.TestCase):
    def setUp(self):
        self.file_name = 'labels.json'
        self.tmp_dir = rv_config.get_tmp_dir()
        self.file_path = os.path.join(self.tmp_dir.name, self.file_name)

        self.crs_transformer = DoubleCRSTransformer()
        self.geojson = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                     [0., 0.]]]
                },
                'properties': {
                    'class_id': 0,
                    'score': 0.9
                }
            }, {
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[1., 1.], [1., 2.], [2., 2.], [2., 1.],
                                     [1., 1.]]]
                },
                'properties': {
                    'score': 0.9,
                    'class_id': 1
                }
            }]
        }

        self.extent = Box.make_square(0, 0, 10)
        self.class_config = ClassConfig(names=['car', 'house'])
        json_to_file(self.geojson, self.file_path)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_read_without_extent(self):
        config = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=self.file_path, default_class_id=None))
        extent = None
        source = config.build(self.class_config, self.crs_transformer, extent,
                              self.tmp_dir.name)
        labels = source.get_labels()

        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        class_ids = np.array([0, 1])
        scores = np.array([0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

    def test_read_with_extent(self):
        # Extent only includes the first box.
        extent = Box.make_square(0, 0, 3)
        config = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=self.file_path, default_class_id=None))
        source = config.build(self.class_config, self.crs_transformer, extent,
                              self.tmp_dir.name)
        labels = source.get_labels()

        npboxes = np.array([[0., 0., 2., 2.]])
        class_ids = np.array([0])
        scores = np.array([0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

        # Extent includes both boxes, but clips the second.
        extent = Box.make_square(0, 0, 3.9)
        config = ObjectDetectionLabelSourceConfig(
            vector_source=GeoJSONVectorSourceConfig(
                uri=self.file_path, default_class_id=None))
        source = config.build(self.class_config, self.crs_transformer, extent,
                              self.tmp_dir.name)
        labels = source.get_labels()

        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 3.9, 3.9]])
        class_ids = np.array([0, 1])
        scores = np.array([0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)


if __name__ == '__main__':
    unittest.main()
