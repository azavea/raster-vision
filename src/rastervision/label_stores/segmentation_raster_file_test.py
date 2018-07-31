import unittest
import numpy as np

from rastervision.core.raster_source import RasterSource
from rastervision.core.box import Box
from rastervision.label_stores.segmentation_raster_file import (
    SegmentationRasterFile)


class TestingRasterSource(RasterSource):
    def __init__(self, zeros=False):
        self.width = 4
        self.height = 4
        self.channels = 3
        if zeros:
            self.data = np.zeros((self.channels, self.height, self.width))
        elif not zeros:
            self.data = np.random.rand(self.channels, self.height, self.width)

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def _get_chip(self, window):
        ymin = window.ymin
        xmin = window.xmin
        ymax = window.ymax
        xmax = window.xmax
        return self.data[:, ymin:ymax, xmin:xmax]

    def get_chip(self, window):
        return self.get_chip(window)

    def get_crs_transformer(self, window):
        return None


class TestSegmentationRasterFile(unittest.TestCase):
    def test_clear(self):
        label_store = SegmentationRasterFile(TestingRasterSource(), None)
        extent = label_store.src.get_extent()
        label_store.clear()
        data = label_store.get_labels(extent)
        self.assertEqual(data.sum(), 0)

    def test_set_labels(self):
        raster_source = TestingRasterSource()
        label_store = SegmentationRasterFile(
            TestingRasterSource(zeros=True), None)
        label_store.set_labels(raster_source)
        extent = label_store.src.get_extent()
        rs_data = raster_source._get_chip(extent)
        ls_data = label_store.get_labels(extent)
        self.assertEqual(rs_data.sum(), ls_data.sum())


if __name__ == '__main__':
    unittest.main()
