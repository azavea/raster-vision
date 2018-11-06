import unittest
import os
import json

from rastervision.data import (ChipClassificationLabelSource,
                               ChipClassificationGeoJSONStore)
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data.label_source.chip_classification_label_source import read_labels
from rastervision.data.label_store.utils import classification_labels_to_geojson
from rastervision.rv_config import RVConfig

from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestChipClassificationGeoJSONStore(unittest.TestCase):
    def setUp(self):
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
                    'class_id': 1
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
                    'class_name': 'house',
                    'class_id': 2
                }
            }]
        }

        self.class_map = ClassMap([ClassItem(1, 'car'), ClassItem(2, 'house')])

        self.file_name = 'labels.json'
        self.temp_dir = RVConfig.get_tmp_dir()
        self.file_path = os.path.join(self.temp_dir.name, self.file_name)

        with open(self.file_path, 'w') as label_file:
            self.geojson_str = json.dumps(self.geojson_dict)
            label_file.write(self.geojson_str)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_classification_labels_to_geojson(self):
        self.maxDiff = None
        extent = Box.make_square(0, 0, 4)
        labels = read_labels(self.geojson_dict, self.crs_transformer, extent)
        geojson_dict = classification_labels_to_geojson(
            labels, self.crs_transformer, self.class_map)
        self.assertDictEqual(geojson_dict, self.geojson_dict)

    def test_constructor_save(self):
        # Read it, write it using label_store, read it again, and compare.
        extent = Box.make_square(0, 0, 10)

        label_source = ChipClassificationLabelSource(
            self.file_path,
            self.crs_transformer,
            self.class_map,
            extent,
            infer_cells=False)

        labels1 = label_source.get_labels()

        new_path = os.path.join(self.temp_dir.name, 'test_save_reload.json')

        label_store = ChipClassificationGeoJSONStore(
            new_path, self.crs_transformer, self.class_map)
        label_store.save(labels1)

        labels2 = label_store.get_labels()

        self.assertDictEqual(labels1.cell_to_class_id,
                             labels2.cell_to_class_id)


if __name__ == '__main__':
    unittest.main()
