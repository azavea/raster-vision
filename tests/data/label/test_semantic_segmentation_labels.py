import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.data.label import SemanticSegmentationLabels


class TestSemanticSegmentationLabels(unittest.TestCase):
    def setUp(self):
        self.windows = [Box.make_square(0, 0, 10), Box.make_square(0, 10, 10)]
        self.label_arr0 = np.random.choice([1, 2], (10, 10))
        self.label_arr1 = np.random.choice([1, 2], (10, 10))

        def label_fn(window):
            if window == self.windows[0]:
                return self.label_arr0.copy()
            elif window == self.windows[1]:
                return self.label_arr1.copy()
            else:
                raise ValueError('Unknown window: {}'.format(window))

        self.label_fn = label_fn
        self.labels = SemanticSegmentationLabels(self.windows, label_fn)

    def test_get(self):
        np.testing.assert_array_equal(
            self.labels.get_label_arr(self.windows[0]), self.label_arr0)

    def test_get_with_aoi(self):
        aoi_polygons = [Box.make_square(5, 15, 2).to_shapely()]
        exp_label_arr = self.label_arr1.copy()
        mask = np.zeros(exp_label_arr.shape)
        mask[5:7, 5:7] = 1
        exp_label_arr = exp_label_arr * mask

        labels = self.labels.filter_by_aoi(aoi_polygons)
        label_arr = labels.get_label_arr(self.windows[1])
        np.testing.assert_array_equal(label_arr, exp_label_arr)

        # Set clip_extent
        clip_extent = Box(0, 0, 10, 18)
        label_arr = labels.get_label_arr(
            self.windows[1], clip_extent=clip_extent)
        np.testing.assert_array_equal(label_arr, exp_label_arr[:, 0:8])


if __name__ == '__main__':
    unittest.main()
