import unittest

from rastervision.core.box import Box
from rastervision.ml_tasks.utils import is_window_inside_aoi


class TestMLTaskUtils(unittest.TestCase):
    def setUp(self):
        self.aoi_polygon = [Box(0, 0, 5, 5).get_shapely()]

    def tearDown(self):
        pass

    def test_is_window_inside_aoi(self):
        test_window1 = Box(1, 2, 1, 2)
        self.assertTrue(is_window_inside_aoi(test_window1, self.aoi_polygon))

        test_window2 = Box(7, 7, 12, 12)
        self.assertFalse(is_window_inside_aoi(test_window2, self.aoi_polygon))

        test_window3 = Box(0, 0, 5, 5)
        self.assertTrue(is_window_inside_aoi(test_window3, self.aoi_polygon))

        self.assertTrue(is_window_inside_aoi(test_window3, None))


if __name__ == '__main__':
    unittest.main()
