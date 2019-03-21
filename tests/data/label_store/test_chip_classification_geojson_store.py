import unittest
import os

import rastervision as rv
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.rv_config import RVConfig
from rastervision.utils.files import json_to_file

from tests.data.mock_crs_transformer import DoubleCRSTransformer


class TestChipClassificationGeoJSONStore(unittest.TestCase):
    def setUp(self):
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

        class MockTaskConfig():
            def __init__(self, class_map):
                self.class_map = class_map

        self.task_config = MockTaskConfig(self.class_map)
        self.temp_dir = RVConfig.get_tmp_dir()
        self.uri = os.path.join(self.temp_dir.name, 'labels.json')

        json_to_file(self.geojson, self.uri)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_constructor_save(self):
        # Read it, write it using label_store, read it again, and compare.
        extent = Box.make_square(0, 0, 10)

        msg = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                .with_uri(self.uri) \
                .build().to_proto()
        config = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                   .from_proto(msg).build()
        label_store = config.create_store(
            self.task_config, extent, self.crs_transformer, self.temp_dir.name)

        labels1 = label_store.get_labels()
        new_uri = os.path.join(self.temp_dir.name, 'test_save_reload.json')
        msg = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                .with_uri(new_uri) \
                .build().to_proto()
        config = rv.LabelStoreConfig.builder(rv.CHIP_CLASSIFICATION_GEOJSON) \
                   .from_proto(msg).build()
        label_store = config.create_store(
            self.task_config, extent, self.crs_transformer, self.temp_dir.name)
        label_store.save(labels1)
        labels2 = label_store.get_labels()

        self.assertDictEqual(labels1.cell_to_class_id,
                             labels2.cell_to_class_id)


if __name__ == '__main__':
    unittest.main()
