import unittest

import numpy as np

from rastervision2.core.box import Box
from rastervision2.core.data.label import SemanticSegmentationLabels


class TestSemanticSegmentationLabels(unittest.TestCase):
    def setUp(self):
        self.windows = [Box.make_square(0, 0, 10), Box.make_square(0, 10, 10)]
        self.label_arr0 = np.random.choice([0, 1], (10, 10))
        self.label_arr1 = np.random.choice([0, 1], (10, 10))
        self.labels = SemanticSegmentationLabels()
        self.labels.set_label_arr(self.windows[0], self.label_arr0)
        self.labels.set_label_arr(self.windows[1], self.label_arr1)

    def test_get(self):
        np.testing.assert_array_equal(
            self.labels.get_label_arr(self.windows[0]), self.label_arr0)

    def test_get_with_aoi(self):
        null_class_id = 2

        aoi_polygons = [Box.make_square(5, 15, 2).to_shapely()]
        exp_label_arr = np.full(self.label_arr1.shape, null_class_id)
        exp_label_arr[5:7, 5:7] = self.label_arr1[5:7, 5:7]

        labels = self.labels.filter_by_aoi(aoi_polygons, null_class_id)
        label_arr = labels.get_label_arr(self.windows[1])
        np.testing.assert_array_equal(label_arr, exp_label_arr)
        self.assertEqual(1, len(labels.window_to_label_arr))


if __name__ == '__main__':
    unittest.main()
