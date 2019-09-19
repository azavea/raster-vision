import unittest

import numpy as np

import rastervision as rv
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from tests.mock import MockRasterSource


class TestSemanticSegmentationLabelSource(unittest.TestCase):
    def test_enough_target_pixels_true(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[4:, 4:, :] = [1, 1, 1]
        raster_source = MockRasterSource([0, 1, 2], 3)
        raster_source.set_raster(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            extent = Box(0, 0, 10, 10)
            self.assertTrue(label_source.enough_target_pixels(extent, 30, [1]))

    def test_enough_target_pixels_false(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = MockRasterSource([0, 1, 2], 3)
        raster_source.set_raster(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            extent = Box(0, 0, 10, 10)
            self.assertFalse(label_source.enough_target_pixels(
                extent, 30, [1]))

    def test_get_labels(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[7:, 7:, 0] = 1
        raster_source = MockRasterSource([0, 1, 2], 3)
        raster_source.set_raster(data)
        label_source = SemanticSegmentationLabelSource(source=raster_source)
        with label_source.activate():
            window = Box.make_square(7, 7, 3)
            labels = label_source.get_labels(window=window)
            label_arr = labels.get_label_arr(window)
            expected_label_arr = np.ones((3, 3))
            np.testing.assert_array_equal(label_arr, expected_label_arr)

    def test_get_labels_rgb(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = MockRasterSource([0, 1, 2], 3)
        raster_source.set_raster(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            window = Box.make_square(7, 7, 3)
            labels = label_source.get_labels(window=window)
            label_arr = labels.get_label_arr(window)
            expected_label_arr = np.ones((3, 3))
            np.testing.assert_array_equal(label_arr, expected_label_arr)

    def test_build_missing(self):
        with self.assertRaises(rv.ConfigError):
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
              .build()

    def test_build(self):
        try:
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
              .with_raster_source('x.tif') \
              .with_rgb_class_map([]) \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_build_deprecated(self):
        try:
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
              .with_raster_source('x.tif') \
              .with_rgb_class_map([]) \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
