import unittest
import os
from tempfile import TemporaryDirectory
import json

import numpy as np

from rastervision.core import Box, ClassMap
from rastervision.data.raster_source import GeoJSONSource
from rastervision.data.crs_transformer import IdentityCRSTransformer
from rastervision.data.utils import boxes_to_geojson
from rastervision.utils.files import str_to_file


class TestGeoJSONSource(unittest.TestCase):
    def setUp(self):
        self.crs_transformer = IdentityCRSTransformer()
        self.extent = Box.make_square(0, 0, 20)
        class_map = ClassMap.construct_from(['car'])
        self.tmp_dir = TemporaryDirectory()

        self.uri = os.path.join(self.tmp_dir.name, 'geo.json')
        boxes = [Box.make_square(5, 5, 10)]
        class_ids = [1]
        geojson = boxes_to_geojson(boxes, class_ids, self.crs_transformer,
                                   class_map)
        str_to_file(json.dumps(geojson), self.uri)

        self.source = GeoJSONSource(self.uri, self.extent,
                                    self.crs_transformer)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_get_extent(self):
        self.assertEqual(self.source.get_extent(), self.extent)

    def test_get_chip(self):
        chip = self.source.get_image_array()
        self.assertEqual(chip.shape, (20, 20, 1))

        expected_chip = 1 * np.ones((20, 20, 1))
        expected_chip[5:15, 5:15, 0] = 2

        np.testing.assert_array_equal(chip, expected_chip)


if __name__ == '__main__':
    unittest.main()
