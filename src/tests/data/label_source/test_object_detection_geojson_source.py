import unittest
import tempfile
import os
import json

import numpy as np
from moto import mock_s3

from rastervision.data.label_source import ObjectDetectionGeoJSONSource
from rastervision.data.label_source.utils import (
    add_classes_to_geojson, geojson_to_object_detection_labels)
from rastervision.data import ObjectDetectionLabels
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.utils.files import NotReadableError

from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestObjectDetectionGeoJSONSource(unittest.TestCase):
    def setUp(self):
        self.mock_s3 = mock_s3()
        self.mock_s3.start()

        self.file_name = 'labels.json'
        self.temp_dir = tempfile.TemporaryDirectory()
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
                    'class_name': 'car',
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
                    'class_name': 'house'
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
                    'class_name': 'car',
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
                    'class_name': 'house'
                }
            }]
        }

        self.linestring_geojson_dict = {
            "type":
            "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {
                    'score': 0.9,
                    'class_name': 'house'
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0., 0.], [0., 1.]]
                }
            }]
        }

        self.extent = Box.make_square(0, 0, 10)
        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        with open(self.file_path, 'w') as label_file:
            self.geojson_str = json.dumps(self.geojson_dict)
            label_file.write(self.geojson_str)

    def tearDown(self):
        self.mock_s3.stop()
        self.temp_dir.cleanup()

    def test_multipolygon_geojson_to_labels(self):
        geojson = add_classes_to_geojson(self.multipolygon_geojson_dict,
                                         self.class_map)
        labels = geojson_to_object_detection_labels(geojson,
                                                    self.crs_transformer)

        # construct expected labels object
        expected_npboxes = np.array([[0., 0., 2., 2.], [2., 2., 4., 4.],
                                     [0., 2., 2., 4.]])
        expected_class_ids = np.array([1, 2, 2])
        expected_scores = np.array([0.9, 0.9, 0.9])
        expected_labels = ObjectDetectionLabels(
            expected_npboxes, expected_class_ids, expected_scores)

        labels.assert_equal(expected_labels)

    def test_polygon_geojson_to_labels(self):
        geojson = add_classes_to_geojson(self.geojson_dict, self.class_map)
        labels = geojson_to_object_detection_labels(geojson,
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
            geojson = add_classes_to_geojson(self.linestring_geojson_dict,
                                             self.class_map)
            geojson_to_object_detection_labels(
                geojson, self.crs_transformer, extent=None)

    def test_read_invalid_uri_readable_true(self):
        with self.assertRaises(NotReadableError):
            invalid_uri = 's3://invalid_path/invalid.json'
            ObjectDetectionGeoJSONSource(
                invalid_uri,
                self.crs_transformer,
                self.class_map,
                extent=self.extent)

    def test_read_without_extent(self):
        store = ObjectDetectionGeoJSONSource(
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
        store = ObjectDetectionGeoJSONSource(
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
        store = ObjectDetectionGeoJSONSource(
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


if __name__ == '__main__':
    unittest.main()
