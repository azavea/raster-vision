import unittest
import tempfile
import os
import json

from rastervision.label_stores.object_detection_geojson_file import (
    ObjectDetectionGeoJSONFile
)
from rastervision.crs_transformers.identity_crs_transformer import (
    IdentityCRSTransformer)
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.utils.files import NotFoundException


class TestObjectDetectionJsonFile(unittest.TestCase):
    def setUp(self):
        self.file_name = 'labels.json'
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_labels(self, labels_dict):
        file_contents = json.dumps(labels_dict)
        with open(self.file_path, 'w') as myfile:
            myfile.write(file_contents)

    def get_valid_labels_dict(self):
        return {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {
                        'label': 'car'
                    },
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates':
                            [ [ [ 0, 0 ],
                                [ 0, 1 ],
                                [ 1, 1 ],
                                [ 1, 0 ],
                                [ 0, 0 ] ] ]
                    }
                }
            ]
        }

    def make_label_store(self, uri, writable):
        crs_transformer = IdentityCRSTransformer()
        extent = Box.make_square(0, 0, 10)
        class_item = ClassItem(1, 'car')
        class_map = ClassMap([class_item])
        return ObjectDetectionGeoJSONFile(
            uri, crs_transformer, extent, class_map, writable)

    def test_open_valid_file(self):
        labels_dict = self.get_valid_labels_dict()
        self.write_labels(labels_dict)
        writable = False

        try:
            self.make_label_store(self.file_path, writable)
        except:
            self.fail(
                'Valid label file raised exception.')

        # TODO test that labels was constructed correctly


if __name__ == '__main__':
    unittest.main()
