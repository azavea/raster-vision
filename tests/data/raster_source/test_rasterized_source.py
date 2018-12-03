import unittest
import os
import json

import numpy as np

import rastervision as rv
from rastervision.core import Box
from rastervision.data.raster_source import RasterSourceConfig
from rastervision.data.crs_transformer import IdentityCRSTransformer
from rastervision.utils.files import str_to_file
from rastervision.rv_config import RVConfig


class TestRasterizedSource(unittest.TestCase):
    def setUp(self):
        self.crs_transformer = IdentityCRSTransformer()
        self.extent = Box.make_square(0, 0, 10)
        self.tmp_dir = RVConfig.get_tmp_dir()
        self.class_id = 2
        self.background_class_id = 3
        self.line_buffer = 1
        self.uri = os.path.join(self.tmp_dir.name, 'temp.json')

    def build_source(self, geojson):
        str_to_file(json.dumps(geojson), self.uri)

        config = RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
            .with_uri(self.uri) \
            .with_rasterizer_options(self.background_class_id, self.line_buffer) \
            .build()

        # Convert to proto and back as a test.
        config = RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
            .from_proto(config.to_proto()) \
            .build()

        source = config.create_source(self.uri, self.crs_transformer,
                                      self.extent)

        return source

    def tearDown(self):
        self.tmp_dir.cleanup()

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
        with source.activate():
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
        with source.activate():
            self.assertEqual(source.get_extent(), self.extent)
            chip = source.get_image_array()
            self.assertEqual(chip.shape, (10, 10, 1))

            expected_chip = self.background_class_id * np.ones((10, 10, 1))
            np.testing.assert_array_equal(chip, expected_chip)


if __name__ == '__main__':
    unittest.main()
