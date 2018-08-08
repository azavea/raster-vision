import unittest
import tempfile
import os
import json

from shapely.geometry import box as ShapelyBox

from rastervision.core.box import Box
from rastervision.core.crs_transformer import CRSTransformer
from rastervision.label_stores.utils import json_to_shapely
from rastervision.ml_tasks.utils import compare_window_to_aoi


class DoubleCRSTransformer(CRSTransformer):
    """Mock CRSTransformer used for testing.

    Assumes map coords are 2x pixels coords.
    """

    def map_to_pixel(self, web_point):
        return (web_point[0] * 2, web_point[1] * 2)

    def pixel_to_map(self, pixel_point):
        return (pixel_point[0] / 2, pixel_point[1] / 2)


class TestMLTaskUtils(unittest.TestCase):
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
                    'coordinates': [[[5., 0.], [5., 5.], [0., 5.], [0., 0.],
                                     [5., 0.]]]
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

    def test_compare_window_to_aoi(self):
        aoi_polygon = json_to_shapely(self.aoi_file_path, self.crs_transformer)
        test_window1 = Box(1, 2, 1, 2)
        self.assertTrue(compare_window_to_aoi(test_window1, aoi_polygon))

        test_window2 = Box(7, 7, 12, 12)
        self.assertFalse(compare_window_to_aoi(test_window2, aoi_polygon))

        test_window3 = Box(0, 0, 10, 10)
        self.assertTrue(compare_window_to_aoi(test_window3, aoi_polygon))


if __name__ == '__main__':
    unittest.main()
