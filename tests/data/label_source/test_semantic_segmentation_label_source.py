import unittest

import numpy as np

import rastervision as rv
from rastervision.core.box import Box
from rastervision.core.class_map import ClassMap, ClassItem
from rastervision.data import (ActivateMixin, ActivationError)
from rastervision.data.label_source.semantic_segmentation_label_source import (
    SemanticSegmentationLabelSource)
from rastervision.data.raster_source.raster_source import RasterSource


class MockRasterSource(ActivateMixin, RasterSource):
    def __init__(self, data):
        self.data = data
        (self.height, self.width, self.channels) = data.shape
        self.activated = False

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def _get_chip(self, window):
        if not self.activated:
            raise ActivationError('MockRasterSource should be activated')

        ymin = window.ymin
        xmin = window.xmin
        ymax = window.ymax
        xmax = window.xmax
        return self.data[ymin:ymax, xmin:xmax, :]

    def get_chip(self, window):
        return self.get_chip(window)

    def get_crs_transformer(self, window):
        return None

    def get_dtype(self):
        return np.uint8

    def _activate(self):
        self.activated = True

    def _deactivate(self):
        self.activated = False


class TestSemanticSegmentationLabelSource(unittest.TestCase):
    def test_enough_target_pixels_true(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[4:, 4:, :] = [1, 1, 1]
        raster_source = MockRasterSource(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            extent = Box(0, 0, 10, 10)
            self.assertTrue(label_source.enough_target_pixels(extent, 30, [1]))

    def test_enough_target_pixels_false(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = MockRasterSource(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            extent = Box(0, 0, 10, 10)
            self.assertFalse(
                label_source.enough_target_pixels(extent, 30, [1]))

    def test_get_labels(self):
        data = np.zeros((10, 10, 1), dtype=np.uint8)
        data[7:, 7:, 0] = 1
        raster_source = MockRasterSource(data)
        label_source = SemanticSegmentationLabelSource(source=raster_source)
        with label_source.activate():
            labels = label_source.get_labels().to_array()
            expected_labels = np.zeros((10, 10))
            expected_labels[7:, 7:] = 1
            np.testing.assert_array_equal(labels, expected_labels)

            window = Box.make_square(7, 7, 3)
            labels = label_source.get_labels(window=window).to_array()
            expected_labels = np.ones((3, 3))
            np.testing.assert_array_equal(labels, expected_labels)

    def test_get_labels_rgb(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = MockRasterSource(data)
        rgb_class_map = ClassMap([ClassItem(id=1, color='#010101')])
        label_source = SemanticSegmentationLabelSource(
            source=raster_source, rgb_class_map=rgb_class_map)
        with label_source.activate():
            labels = label_source.get_labels().to_array()
            expected_labels = np.zeros((10, 10))
            expected_labels[7:, 7:] = 1
            np.testing.assert_array_equal(labels, expected_labels)

            window = Box.make_square(7, 7, 3)
            labels = label_source.get_labels(window=window).to_array()
            expected_labels = np.ones((3, 3))
            np.testing.assert_array_equal(labels, expected_labels)

    def test_build_missing(self):
        with self.assertRaises(rv.ConfigError):
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
              .build()

    def test_build(self):
        try:
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
              .with_raster_source('x.geojson') \
              .with_rgb_class_map([]) \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')

    def test_build_deprecated(self):
        try:
            rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION_RASTER) \
              .with_raster_source('x.geojson') \
              .with_rgb_class_map([]) \
              .build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
