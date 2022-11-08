import unittest

import numpy as np

from rastervision.core import Box
from rastervision.core.data import (
    ClassConfig, SemanticSegmentationLabelSource, RGBClassTransformer)
from tests.core.data.mock_raster_source import MockRasterSource


class TestSemanticSegmentationLabelSourceConfig(unittest.TestCase):
    pass


class TestSemanticSegmentationLabelSource(unittest.TestCase):
    def test_enough_target_pixels_true(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[4:, 4:, :] = 1
        class_config = ClassConfig(names=['bg', 'fg', 'null'])
        raster_source = MockRasterSource([0], 1)
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(raster_source,
                                                       class_config)
        extent = Box(0, 0, 10, 10)
        self.assertTrue(label_source.enough_target_pixels(extent, 30, [1]))

    def test_enough_target_pixels_false(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[7:, 7:, :] = 1
        class_config = ClassConfig(names=['bg', 'fg', 'null'])
        raster_source = MockRasterSource([0], 1)
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(raster_source,
                                                       class_config)
        extent = Box(0, 0, 10, 10)
        self.assertFalse(label_source.enough_target_pixels(extent, 30, [1]))

    def test_get_labels(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[7:, 7:, 0] = 1
        class_config = ClassConfig(names=['bg', 'fg', 'null'])
        raster_source = MockRasterSource([0], 1)
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(raster_source,
                                                       class_config)
        window = Box.make_square(7, 7, 3)
        labels = label_source.get_labels(window=window)
        label_arr = labels.get_label_arr(window)
        expected_label_arr = np.ones((3, 3))
        np.testing.assert_array_equal(label_arr, expected_label_arr)

    def test_get_label_arr_off_edge(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[7:, 7:, 0] = 1
        class_config = ClassConfig(names=['bg', 'fg', 'null'])
        raster_source = MockRasterSource([0], 1)
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(raster_source,
                                                       class_config)
        window = Box.make_square(7, 7, 6)
        label_arr = label_source.get_label_arr(window)
        expected_label_arr = np.full((6, 6), class_config.null_class_id)
        expected_label_arr[0:3, 0:3] = 1
        np.testing.assert_array_equal(label_arr, expected_label_arr)

    def test_get_labels_rgb(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        class_config = ClassConfig(names=['bg', 'fg', 'null'])
        rgb_class_config = ClassConfig(names=['a'], colors=['#010101'])
        rgb_class_config.ensure_null_class()
        raster_source = MockRasterSource(
            [0, 1, 2],
            3,
            raster_transformers=[
                RGBClassTransformer(class_config=rgb_class_config)
            ])
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(raster_source,
                                                       class_config)
        window = Box.make_square(7, 7, 3)
        labels = label_source.get_labels(window=window)
        label_arr = labels.get_label_arr(window)
        expected_label_arr = np.zeros((3, 3))
        np.testing.assert_array_equal(label_arr, expected_label_arr)


if __name__ == '__main__':
    unittest.main()
