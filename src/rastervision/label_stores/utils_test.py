import unittest
import tempfile
import os
import json

from rastervision.label_stores.utils import json_to_shapely
from rastervision.core.crs_transformer import CRSTransformer
from rastervision.core.box import Box


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes map coords are 2x pixels coords.
    """

    def map_to_pixel(self, web_point):
        return (web_point[0] * 2, web_point[1] * 2)

    def pixel_to_map(self, pixel_point):
        return (pixel_point[0] / 2, pixel_point[1] / 2)


class TestLabelStoreUtils(unittest.TestCase):

    def setUp(self):
        self.crs_transformer = DoubleCRSTransformer()
        self.aoi_dict = {
            'type':
            'FeatureCollection',
            'features': [{
                'type': 'Feature',
                'geometry': {
                    'type':
                    'Polygon',
                    'coordinates': [[[1., 0.], [1., 1.], [0., 1.], [0., 0.],
                                     [1., 0.]]]
                },
                'properties': {}
            }]
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        self.aoi_file_name = 'aoi.json'
        self.aoi_file_path = os.path.join(self.temp_dir.name,
                                          self.aoi_file_name)

        with open(self.aoi_file_path, 'w') as aoi_file:
            self.aoi_str = json.dumps(self.aoi_dict)
            aoi_file.write(self.aoi_str)

    def tearDown(self):
        pass

    def test_json_to_shapely(self):
        self.assertIsNone(json_to_shapely(None, self.crs_transformer))

        aoi_polygon = json_to_shapely(self.aoi_dict, self.crs_transformer)
        aoi_box = Box.make_square(0, 0, 2)

        self.assertEqual(len(aoi_polygon), 1)
        self.assertTrue(aoi_polygon[0].equals(aoi_box.get_shapely()))


if __name__ == '__main__':
    unittest.main()
