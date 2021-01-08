import unittest

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.label import (SemanticSegmentationLabels,
                                          SemanticSegmentationDiscreteLabels,
                                          SemanticSegmentationSmoothLabels)


class TestSemanticSegmentationLabels(unittest.TestCase):
    def test_build(self):
        self.assertIsInstance(
            SemanticSegmentationLabels.build(smooth=False),
            SemanticSegmentationDiscreteLabels)

        self.assertRaises(
            ValueError, lambda: SemanticSegmentationLabels.build(smooth=True))

        self.assertRaises(
            ValueError, lambda: SemanticSegmentationLabels.build(
                smooth=True, extent=Box(0, 0, 10, 10)))

        self.assertIsInstance(
            SemanticSegmentationLabels.build(
                smooth=True, extent=Box(0, 0, 10, 10), num_classes=2),
            SemanticSegmentationSmoothLabels)


class TestSemanticSegmentationDiscreteLabels(unittest.TestCase):
    def setUp(self):
        self.windows = [Box.make_square(0, 0, 10), Box.make_square(0, 10, 10)]
        self.label_arr0 = np.random.choice([0, 1], (10, 10))
        self.label_arr1 = np.random.choice([0, 1], (10, 10))
        self.labels = SemanticSegmentationDiscreteLabels()
        self.labels[self.windows[0]] = self.label_arr0
        self.labels[self.windows[1]] = self.label_arr1

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


def make_random_scores(num_classes, h, w):
    arr = np.random.normal(size=(num_classes, h, w))
    # softmax
    arr = np.exp(arr, out=arr)
    arr /= arr.sum(axis=0)
    return arr


class TestSemanticSegmentationSmoothLabels(unittest.TestCase):
    def setUp(self):
        self.windows = [
            Box.make_square(0, 0, 10),
            Box.make_square(0, 5, 10),
            Box.make_square(0, 10, 10)
        ]
        self.num_classes = 3
        self.scores_left = make_random_scores(self.num_classes, 10, 10)
        self.scores_mid = make_random_scores(self.num_classes, 10, 10)
        self.scores_right = make_random_scores(self.num_classes, 10, 10)

        self.scores_left = self.scores_left.astype(np.float16)
        self.scores_mid = self.scores_mid.astype(np.float16)
        self.scores_right = self.scores_right.astype(np.float16)

        self.extent = Box(0, 0, 10, 20)
        self.labels = SemanticSegmentationSmoothLabels(
            extent=self.extent, num_classes=self.num_classes)
        self.labels[self.windows[0]] = self.scores_left
        self.labels[self.windows[1]] = self.scores_mid
        self.labels[self.windows[2]] = self.scores_right

        arr = np.zeros((self.num_classes, 10, 20), dtype=np.float16)
        arr[..., :10] += self.scores_left
        arr[..., 5:15] += self.scores_mid
        arr[..., 10:] += self.scores_right
        self.expected_scores = arr

        hits = np.zeros((10, 20), dtype=np.uint8)
        hits[..., :10] += 1
        hits[..., 5:15] += 1
        hits[..., 10:] += 1
        self.expected_hits = hits

    def test_pixel_scores(self):
        np.testing.assert_array_almost_equal(self.expected_scores,
                                             self.labels.pixel_scores)

    def test_get_scores_arr(self):
        avg_scores = self.expected_scores / self.expected_hits
        np.testing.assert_array_almost_equal(
            avg_scores, self.labels.get_score_arr(self.extent))

    def test_get_label_arr(self):
        avg_scores = self.expected_scores / self.expected_hits
        labels = np.argmax(avg_scores, axis=0)
        np.testing.assert_array_equal(labels,
                                      self.labels.get_label_arr(self.extent))

    def test_pixel_hits(self):
        np.testing.assert_array_equal(self.expected_hits,
                                      self.labels.pixel_hits)

    def test_eq(self):
        labels = SemanticSegmentationSmoothLabels(
            extent=self.extent, num_classes=self.num_classes)
        labels.pixel_hits = self.expected_hits
        labels.pixel_scores = self.expected_scores
        self.assertTrue(labels == self.labels)

    def test_get_with_aoi(self):
        null_class_id = 2

        aoi = Box.make_square(5, 15, 2)
        aoi_polygons = [aoi.to_shapely()]
        exp_label_arr = self.labels.get_label_arr(self.windows[2])
        exp_label_arr[:] = null_class_id
        y0, x0, y1, x1 = aoi
        x0, x1 = x0 - 10, x1 - 10
        exp_label_arr[y0:y1, x0:x1] = self.labels.get_label_arr(aoi)

        labels = self.labels.filter_by_aoi(aoi_polygons, null_class_id)
        label_arr = labels.get_label_arr(self.windows[2])
        np.testing.assert_array_equal(label_arr, exp_label_arr)


if __name__ == '__main__':
    unittest.main()
