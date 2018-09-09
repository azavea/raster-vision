import unittest
import numpy as np

from rastervision.core.raster_source import RasterSource
from rastervision.core.box import Box
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationRasterFile)


class TestingRasterSource(RasterSource):
    def __init__(self, zeros=False, data=None):
        if data is not None:
            self.data = data
        elif zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.zeros(
                (self.height, self.width, self.channels), dtype=np.uint8)
        elif not zeros and data is None:
            self.width = 4
            self.height = 4
            self.channels = 3
            self.data = np.random.randint(
                0,
                2,
                size=(self.width, self.height, self.channels),
                dtype=np.uint8)
            self.data[:, :, 0:(self.channels - 1)] = np.zeros(
                (self.height, self.width, self.channels - 1), dtype=np.uint8)

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def _get_chip(self, window):
        ymin = window.ymin
        xmin = window.xmin
        ymax = window.ymax
        xmax = window.xmax
        return self.data[ymin:ymax, xmin:xmax, :]

    def get_chip(self, window):
        return self.get_chip(window)

    def get_crs_transformer(self, window):
        return None


class TestSegmentationRasterFile(unittest.TestCase):
    def test_clear(self):
        label_store = SegmentationRasterFile(TestingRasterSource(), None, None)
        extent = label_store.source.get_extent()
        label_store.clear()
        data = label_store.get_labels(extent)
        self.assertEqual(data.sum(), 0)

    def test_set_labels(self):
        raster_source = TestingRasterSource(zeros=True)
        label_store = SegmentationRasterFile(
            source=raster_source,
            sink=None,
            class_map=None,
            raster_class_map={'#000001': 1})
        label_store.set_labels(raster_source)
        extent = label_store.source.get_extent()
        rs_data = raster_source._get_chip(extent)
        ls_data = (label_store.get_labels(extent) == 1)
        self.assertEqual(rs_data.sum(), ls_data.sum())

    def test_window_predicate_true(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[4:, 4:, :] = [1, 1, 1]
        raster_source = TestingRasterSource(data=data)
        label_store = SegmentationRasterFile(
            source=raster_source,
            sink=None,
            class_map=None,
            raster_class_map={'#010101': 1})
        extent = Box(0, 0, 10, 10)
        self.assertTrue(label_store.window_predicate(extent, [1]))

    def test_window_predicate_false(self):
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        data[7:, 7:, :] = [1, 1, 1]
        raster_source = TestingRasterSource(data=data)
        label_store = SegmentationRasterFile(
            source=raster_source,
            sink=None,
            class_map=None,
            raster_class_map={'#010101': 1})
        extent = Box(0, 0, 10, 10)
        self.assertFalse(label_store.window_predicate(extent, [1]))


if __name__ == '__main__':
    unittest.main()
