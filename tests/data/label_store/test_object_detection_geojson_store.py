import unittest
import os
import json

from moto import mock_s3

from rastervision.data import ObjectDetectionGeoJSONStore, ObjectDetectionLabelSource
from rastervision.data.label_source.utils import (
    geojson_to_object_detection_labels)
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.filesystem import NotWritableError
from rastervision.rv_config import RVConfig

from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestObjectDetectionLabelSource(unittest.TestCase):
    def setUp(self):
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
                    'class_id': 1
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
        self.temp_dir.cleanup()

    @mock_s3
    def test_write_invalid_uri(self):
        labels = geojson_to_object_detection_labels(self.geojson_dict,
                                                    self.crs_transformer)

        invalid_uri = 's3://invalid_path/invalid.json'
        label_store = ObjectDetectionGeoJSONStore(
            invalid_uri, self.crs_transformer, self.class_map)
        with self.assertRaises(NotWritableError):
            label_store.save(labels)

    def test_valid_uri(self):
        # Read it, write it using label_store, read it again, and compare.
        label_source = ObjectDetectionLabelSource(
            self.file_path, self.crs_transformer, self.class_map, self.extent)
        labels1 = label_source.get_labels()

        new_path = os.path.join(self.temp_dir.name, 'test_save_reload.json')

        label_store = ObjectDetectionGeoJSONStore(
            new_path, self.crs_transformer, self.class_map)
        label_store.save(labels1)

        label_store = ObjectDetectionLabelSource(
            self.file_path, self.crs_transformer, self.class_map, self.extent)
        labels2 = label_store.get_labels()

        labels1.assert_equal(labels2)


if __name__ == '__main__':
    unittest.main()
