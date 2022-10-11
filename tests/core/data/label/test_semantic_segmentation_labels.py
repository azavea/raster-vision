import unittest
from os.path import join

import numpy as np
import rasterio as rio

from rastervision.pipeline import rv_config
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassConfig, IdentityCRSTransformer, SemanticSegmentationLabels,
    SemanticSegmentationDiscreteLabels, SemanticSegmentationSmoothLabels)


class TestSemanticSegmentationLabels(unittest.TestCase):
    def test_build(self):
        extent = Box(0, 0, 10, 10)
        num_classes = 2

        # smooth=False
        msg = ('smooth=False should return a '
               'SemanticSegmentationDiscreteLabels instance')
        labels = SemanticSegmentationLabels.make_empty(
            extent=extent, num_classes=num_classes, smooth=False)
        self.assertIsInstance(
            labels, SemanticSegmentationDiscreteLabels, msg=msg)

        # smooth=True
        msg = ('smooth=True should return a '
               'SemanticSegmentationSmoothLabels instance')
        labels = SemanticSegmentationLabels.make_empty(
            extent=extent, num_classes=num_classes, smooth=True)
        self.assertIsInstance(
            labels, SemanticSegmentationSmoothLabels, msg=msg)

    def test_from_predictions_smooth(self):
        def make_pred_chip() -> np.ndarray:
            chip = np.concatenate([
                np.random.uniform(0.001, 0.5, size=(1, 40, 40)),
                np.random.uniform(0.5, 1, size=(1, 40, 40))
            ])
            chip[1, 10:-10, 10:-10] = 0
            chip /= chip.sum(axis=0)
            return chip

        extent = Box(0, 0, 80, 80)
        windows = extent.get_windows(40, stride=20, padding=0)

        predictions = [make_pred_chip() for _ in windows]
        labels = SemanticSegmentationLabels.from_predictions(
            windows,
            predictions,
            extent=extent,
            num_classes=2,
            smooth=True,
            crop_sz=10)
        label_arr = labels.get_label_arr(extent)

        exp_label_arr = np.full(extent.size, -1)
        exp_label_arr[10:-10, 10:-10] = 0
        np.testing.assert_array_equal(label_arr, exp_label_arr)

    def test_from_predictions_discrete(self):
        def make_pred_chip_labels() -> np.ndarray:
            chip = np.full((40, 40), 1)
            chip[10:-10, 10:-10] = 0
            return chip

        extent = Box(0, 0, 80, 80)
        windows = extent.get_windows(40, stride=20, padding=0)

        exp_label_arr = np.full(extent.size, -1)
        exp_label_arr[10:-10, 10:-10] = 0

        predictions = [make_pred_chip_labels() for _ in windows]
        labels = SemanticSegmentationLabels.from_predictions(
            windows,
            predictions,
            extent=extent,
            num_classes=2,
            smooth=False,
            crop_sz=10)
        label_arr = labels.get_label_arr(extent)

        exp_label_arr = np.full(extent.size, -1)
        exp_label_arr[10:-10, 10:-10] = 0
        np.testing.assert_array_equal(label_arr, exp_label_arr)


class TestSemanticSegmentationDiscreteLabels(unittest.TestCase):
    def setUp(self):
        self.class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        num_classes = len(self.class_config)
        extent = Box(0, 0, 20, 20)
        self.windows = [Box.make_square(0, 0, 10), Box.make_square(0, 10, 10)]
        self.label_arr0 = np.random.randint(0, num_classes, size=(10, 10))
        self.label_arr1 = np.random.randint(0, num_classes, size=(10, 10))
        self.labels = SemanticSegmentationDiscreteLabels(
            extent=extent, num_classes=num_classes)
        self.labels[self.windows[0]] = self.label_arr0
        self.labels[self.windows[1]] = self.label_arr1

    def test_get_label_arr(self):
        np.testing.assert_array_equal(
            self.labels.get_label_arr(self.windows[0]), self.label_arr0)

    def test_get_label_arr_empty(self):
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationDiscreteLabels(extent, 2)
        label_arr = labels.get_label_arr(extent)
        np.testing.assert_array_equal(label_arr, np.full((3, 3), -1))

        window = Box(0, 0, 2, 2)
        labels[window] = np.ones((2, 2))
        label_arr = labels.get_label_arr(extent)
        exp_label_arr = np.array([
            [1, 1, -1],
            [1, 1, -1],
            [-1, -1, -1],
        ])
        np.testing.assert_array_equal(label_arr, exp_label_arr)

    def test_get_with_aoi(self):
        null_class_id = self.class_config.null_class_id

        aoi_polygons = [Box.make_square(5, 15, 2).to_shapely()]
        exp_label_arr = np.full_like(self.label_arr1, fill_value=null_class_id)
        exp_label_arr[5:7, 5:7] = self.label_arr1[5:7, 5:7]

        labels = self.labels.filter_by_aoi(aoi_polygons, null_class_id)
        label_arr = labels.get_label_arr(self.windows[1])
        np.testing.assert_array_equal(label_arr, exp_label_arr)

    def test_make_empty(self):
        extent = Box(0, 0, 10, 10)
        num_classes = 3
        labels = SemanticSegmentationDiscreteLabels.make_empty(
            extent=extent, num_classes=num_classes)
        self.assertEqual(labels.extent, extent)
        self.assertEqual(labels.num_classes, num_classes)
        self.assertEqual(labels.dtype, np.uint8)
        self.assertEqual(labels.pixel_counts.shape,
                         (num_classes, *extent.size))

    def test_setitem(self):
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationDiscreteLabels(extent, 2)

        labels[extent] = np.eye(3)
        labels[extent] = np.eye(3)
        labels[extent] = np.random.randint(0, 2, size=(3, 3))
        label_arr = labels.get_label_arr(extent)
        np.testing.assert_array_equal(label_arr, np.eye(3))

    def test_delitem(self):
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationDiscreteLabels(extent, 2)
        labels[extent] = np.eye(3)
        del labels[extent]
        label_arr = labels.get_label_arr(extent)
        np.testing.assert_array_equal(label_arr, np.full((3, 3), -1))

    def test_eq(self):
        extent = Box(0, 0, 3, 3)
        labels1 = SemanticSegmentationDiscreteLabels(extent, 2)
        labels1[extent] = np.eye(3)
        labels2 = SemanticSegmentationDiscreteLabels(extent, 2)
        labels2[extent] = np.eye(3)
        labels3 = SemanticSegmentationDiscreteLabels(extent, 2)
        labels3[extent] = np.zeros((3, 3))
        self.assertEqual(labels1, labels2)
        self.assertNotEqual(labels1, labels3)
        self.assertNotEqual(labels2, labels3)

    def test_get_score_arr(self):
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationDiscreteLabels(extent, 2)

        labels[extent] = np.zeros((3, 3))
        labels[extent] = np.ones((3, 3))
        scores = labels.get_score_arr(extent)
        np.testing.assert_array_equal(scores[0], np.full((3, 3), 0.5))
        np.testing.assert_array_equal(scores[1], np.full((3, 3), 0.5))

        labels[extent] = np.ones((3, 3))
        labels[extent] = np.ones((3, 3))
        scores = labels.get_score_arr(extent)
        np.testing.assert_array_equal(scores[0], np.full((3, 3), 0.25))
        np.testing.assert_array_equal(scores[1], np.full((3, 3), 0.75))

    def test_save(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationDiscreteLabels(extent, 2)
        labels[extent] = np.eye(3)
        exp_arr = labels.get_label_arr(extent)

        with rv_config.get_tmp_dir() as tmp_dir:
            uri = join(tmp_dir, 'test')
            labels.save(
                uri=uri,
                crs_transformer=IdentityCRSTransformer(),
                class_config=class_config)
            with rio.open(join(uri, 'labels.tif'), 'r') as ds:
                arr = ds.read(1)
                np.testing.assert_array_equal(arr, exp_arr)


def make_random_scores(num_classes, h, w):
    arr = np.random.normal(size=(num_classes, h, w))
    # softmax
    arr = np.exp(arr, out=arr)
    arr /= arr.sum(axis=0)
    return arr.astype(np.float16)


class TestSemanticSegmentationSmoothLabels(unittest.TestCase):
    def setUp(self):
        self.extent = Box(0, 0, 10, 20)
        self.num_classes = 3
        self.windows = [
            Box(0, 0, 10, 10),
            Box(0, 5, 10, 15),
            Box(0, 10, 10, 20)
        ]
        self.scores_left = make_random_scores(self.num_classes, 10, 10)
        self.scores_mid = make_random_scores(self.num_classes, 10, 10)
        self.scores_right = make_random_scores(self.num_classes, 10, 10)

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
        exp_label_arr = np.argmax(avg_scores, axis=0)
        label_arr = self.labels.get_label_arr(self.extent)
        np.testing.assert_array_equal(label_arr, exp_label_arr)

    def test_get_label_arr_empty(self):
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationSmoothLabels(extent, 2)
        label_arr = labels.get_label_arr(extent)
        np.testing.assert_array_equal(label_arr, np.full((3, 3), -1))

        window = Box(0, 0, 2, 2)
        labels[window] = np.random.random(size=(2, 2))
        label_arr = labels.get_label_arr(extent)
        np.testing.assert_array_equal(label_arr[:, 2], np.array([-1, -1, -1]))
        np.testing.assert_array_equal(label_arr[2, :], np.array([-1, -1, -1]))

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

        aoi = Box(5, 15, 7, 17)
        aoi_polygons = [aoi.to_shapely()]
        exp_label_arr = self.labels.get_label_arr(self.windows[2])
        exp_label_arr[:] = null_class_id
        y0, x0, y1, x1 = aoi
        x0, x1 = x0 - 10, x1 - 10
        exp_label_arr[y0:y1, x0:x1] = self.labels.get_label_arr(aoi)

        labels = self.labels.filter_by_aoi(aoi_polygons, null_class_id)
        label_arr = labels.get_label_arr(self.windows[2])
        np.testing.assert_array_equal(label_arr, exp_label_arr)

    def test_to_local_coords(self):
        extent = Box(100, 100, 200, 200)
        labels = SemanticSegmentationSmoothLabels(extent=extent, num_classes=2)

        # normal window
        box_in = (120, 150, 170, 200)
        box_out_expected = (20, 50, 70, 100)
        box_out_actual = labels._to_local_coords(box_in)
        self.assertTupleEqual(box_out_actual, box_out_expected)

        # window completely outside the extent
        box_in = (0, 0, 100, 100)
        box_out_expected = (0, 0, 0, 0)
        box_out_actual = labels._to_local_coords(box_in)
        self.assertTupleEqual(box_out_actual, box_out_expected)

        # window completely outside the extent
        box_in = (200, 200, 300, 300)
        box_out_expected = (100, 100, 100, 100)
        box_out_actual = labels._to_local_coords(box_in)
        self.assertTupleEqual(box_out_actual, box_out_expected)

    def test_save(self):
        class_config = ClassConfig(names=['bg', 'fg'], null_class='bg')
        extent = Box(0, 0, 3, 3)
        labels = SemanticSegmentationSmoothLabels(extent, 2)
        labels[extent] = np.random.random(extent.size)
        exp_label_arr = labels.get_label_arr(extent, null_class_id=0)
        exp_score_arr = labels.get_score_arr(extent)
        exp_hits_arr = labels.pixel_hits

        with rv_config.get_tmp_dir() as tmp_dir:
            uri = join(tmp_dir, 'test')
            labels.save(
                uri=uri,
                crs_transformer=IdentityCRSTransformer(),
                class_config=class_config)

            with rio.open(join(uri, 'labels.tif'), 'r') as ds:
                arr = ds.read(1)
                np.testing.assert_array_equal(arr, exp_label_arr)

            with rio.open(join(uri, 'scores.tif'), 'r') as ds:
                arr = ds.read()
                np.testing.assert_array_equal(arr, exp_score_arr)

            hits_arr = np.load(join(uri, 'pixel_hits.npy'))
            np.testing.assert_array_equal(hits_arr, exp_hits_arr)


if __name__ == '__main__':
    unittest.main()
