import unittest
import os
import json

import numpy as np
from moto import mock_s3

import rastervision as rv

from rastervision.data.label_source import ObjectDetectionLabelSource
from rastervision.data.label_source.utils import (
    geojson_to_object_detection_labels)
from rastervision.data import ObjectDetectionLabels
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.filesystem import NotReadableError
from rastervision.rv_config import RVConfig
from rastervision.data.crs_transformer import IdentityCRSTransformer

from tests import data_file_path
from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestObjectDetectionLabelSource(unittest.TestCase):
    def setUp(self):
        self.prev_keys = (os.environ.get('AWS_ACCESS_KEY_ID'),
                          os.environ.get('AWS_SECRET_ACCESS_KEY'))
        os.environ['AWS_ACCESS_KEY_ID'] = 'DUMMY'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'DUMMY'
        self.mock_s3 = mock_s3()
        self.mock_s3.start()

        self.file_name = 'labels.json'
        self.temp_dir = RVConfig.get_tmp_dir()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

        self.crs_transformer = DoubleCRSTransformer()
        self.geojson_dict = {
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
                    'class_id': 1,
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
                    'class_id': 2
                }
            }]
        }

        self.multipolygon_geojson_dict = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'MultiPolygon',
                    'coordinates': [[[[0., 0.], [0., 1.], [1., 1.], [1., 0.],
                                      [0., 0.]]]]
                },
                'properties': {
                    'class_id': 1,
                    'score': 0.9
                }
            }, {
                'type': 'Feature',
                'geometry': {
                    'type':
                    'MultiPolygon',
                    'coordinates':
                    [[[[1., 1.], [1., 2.], [2., 2.], [2., 1.], [1., 1.]]],
                     [[[1., 0.], [1., 1.], [2., 1.], [2., 0.], [1., 0.]]]]
                },
                'properties': {
                    'score': 0.9,
                    'class_id': 2
                }
            }]
        }

        self.linestring_geojson_dict = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'properties': {
                    'score': 0.9,
                    'class_id': 2
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[0., 0.], [0., 1.]]
                }
            }]
        }

        self.extent = Box.make_square(0, 0, 10)
        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        with open(self.file_path, 'w') as label_file:
            self.geojson_str = json.dumps(self.geojson_dict)
            label_file.write(self.geojson_str)

    def tearDown(self):
        access, secret = self.prev_keys
        if access:
            os.environ['AWS_ACCESS_KEY_ID'] = access
        else:
            del os.environ['AWS_ACCESS_KEY_ID']
        if secret:
            os.environ['AWS_SECRET_ACCESS_KEY'] = secret
        else:
            del os.environ['AWS_SECRET_ACCESS_KEY']

        self.mock_s3.stop()
        self.temp_dir.cleanup()

    def test_multipolygon_geojson_to_labels(self):
        labels = geojson_to_object_detection_labels(
            self.multipolygon_geojson_dict, self.crs_transformer)

        # construct expected labels object
        expected_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.],
                                     [0., 2., 2., 4.]])
        expected_class_ids = np.array([1, 2, 2])
        expected_scores = np.array([0.9, 0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            expected_npboxes, expected_class_ids, expected_scores)

        labels.assert_equal(expected_labels)

    def test_polygon_geojson_to_labels(self):
        labels = geojson_to_object_detection_labels(self.geojson_dict,
                                                    self.crs_transformer)

        # construct expected labels object
        expected_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        expected_class_ids = np.array([1, 2])
        expected_scores = np.array([0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            expected_npboxes, expected_class_ids, expected_scores)

        labels.assert_equal(expected_labels)

    def test_read_invalid_geometry_type(self):
        with self.assertRaises(Exception):
            geojson_to_object_detection_labels(
                self.linestring_geojson_dict,
                self.crs_transformer,
                extent=None)

    def test_read_invalid_uri_readable_true(self):
        with self.assertRaises(NotReadableError):
            invalid_uri = 's3://invalid_path/invalid.json'
            ObjectDetectionLabelSource(
                invalid_uri,
                self.crs_transformer,
                self.class_map,
                extent=self.extent)

    def test_read_without_extent(self):
        store = ObjectDetectionLabelSource(
            self.file_path, self.crs_transformer, self.class_map, extent=None)
        labels = store.get_labels()

        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.]])
        class_ids = np.array([1, 2])
        scores = np.array([0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

    def test_read_with_extent(self):
        # Extent only includes the first box.
        extent = Box.make_square(0, 0, 3)
        store = ObjectDetectionLabelSource(
            self.file_path,
            self.crs_transformer,
            self.class_map,
            extent=extent)
        labels = store.get_labels()

        npboxes = np.array([[0., 0., 2., 2.]])
        class_ids = np.array([1])
        scores = np.array([0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

        # Extent includes both boxes, but clips the second.
        extent = Box.make_square(0, 0, 3.9)
        store = ObjectDetectionLabelSource(
            self.file_path,
            self.crs_transformer,
            self.class_map,
            extent=extent)
        labels = store.get_labels()

        npboxes = np.array([[0., 0., 2., 2.], [2., 2., 3.9, 3.9]])
        class_ids = np.array([1, 2])
        scores = np.array([0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            npboxes, class_ids, scores=scores)
        labels.assert_equal(expected_labels)

    def test_missing_config_uri(self):
        with self.assertRaises(rv.ConfigError):
            rv.data.ObjectDetectionLabelSourceConfig.builder(
                rv.OBJECT_DETECTION).build()

    def test_no_missing_config(self):
        try:
            rv.data.ObjectDetectionLabelSourceConfig.builder(
                rv.OBJECT_DETECTION).with_uri('x.geojson').build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_deprecated_builder(self):
        try:
            rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION_GEOJSON) \
              .with_uri('x.geojson') \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_builder(self):
        uri = data_file_path('polygon-labels.geojson')
        msg = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                .with_vector_source(uri) \
                .build().to_proto()
        config = rv.LabelSourceConfig.builder(rv.OBJECT_DETECTION) \
                   .from_proto(msg).build()
        self.assertEqual(config.vector_source.uri, uri)

        classes = ['one', 'two']
        extent = Box.make_square(0, 0, 10)
        crs_transformer = IdentityCRSTransformer()
        with RVConfig.get_tmp_dir() as tmp_dir:
            task_config = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_classes(classes) \
                            .build()
            config.create_source(task_config, extent, crs_transformer, tmp_dir)


if __name__ == '__main__':
    unittest.main()
