import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.data.label import SemanticSegmentationLabels


class TestSemanticSegmentationLabels(unittest.TestCase):
    def setUp(self):
        labels = np.zeros((100, 100))
        # Make 2x2 grid that spans 200x200
        # yapf: disable
        label_pairs = [
            (Box.make_square(0, 0, 100), labels),
            (Box.make_square(0, 100, 100), labels),
            (Box.make_square(100, 0, 100), labels),
            (Box.make_square(100, 100, 100), labels)
        ]
        # yapf: enable
        self.labels = SemanticSegmentationLabels(label_pairs)

    def test_get_extent(self):
        extent = self.labels.get_extent()
        self.assertTrue(extent == Box.make_square(0, 0, 200))

    def test_get_clipped_labels(self):
        extent = Box.make_square(0, 0, 150)
        clipped = self.labels.get_clipped_labels(extent)
        pairs = clipped.get_label_pairs()
        # yapf: disable
        expected_pairs = [
            (Box.make_square(0, 0, 100), np.zeros((100, 100))),
            (Box(0, 100, 100, 150), np.zeros((100, 50))),
            (Box(100, 0, 150, 100), np.zeros((50, 100))),
            (Box(100, 100, 150, 150), np.zeros((50, 50)))
        ]
        # yapf: enable

        def to_tuple(label_pairs):
            return [(p[0].tuple_format(), p[1].shape) for p in label_pairs]

        pairs = to_tuple(pairs)
        expected_pairs = to_tuple(expected_pairs)
        for pair, expected_pair in zip(pairs, expected_pairs):
            self.assertTupleEqual(pair, expected_pair)

    def test_from_array(self):
        arr = np.zeros((5, 5))
        labels = SemanticSegmentationLabels.from_array(arr)
        pairs = labels.get_label_pairs()
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0], Box.make_square(0, 0, 5))
        np.testing.assert_array_equal(pairs[0][1], arr)

    def test_to_array(self):
        arr = self.labels.to_array()
        expected_arr = np.zeros((200, 200))
        np.testing.assert_array_equal(arr, expected_arr)

    def test_filter_by_aoi(self):
        arr = np.ones((5, 5))
        labels = SemanticSegmentationLabels.from_array(arr)
        aoi_polygons = [Box.make_square(0, 0, 2).to_shapely()]
        labels = labels.filter_by_aoi(aoi_polygons)
        arr = labels.to_array()
        expected_arr = np.zeros((5, 5))
        expected_arr[0:2, 0:2] = 1
        np.testing.assert_array_equal(arr, expected_arr)


if __name__ == '__main__':
    unittest.main()
