from unittest.mock import Mock
import numpy as np

from rastervision.core import Box
from rastervision.core.data import RasterSource, IdentityCRSTransformer


class MockRasterSource(RasterSource):
    def __init__(self, channel_order, num_channels_raw,
                 raster_transformers=[]):
        super().__init__(channel_order, num_channels_raw, raster_transformers)
        self.mock = Mock()
        self.set_return_vals()

    def set_return_vals(self, raster=None):
        self._dtype = np.uint8
        self._extent = Box.make_square(0, 0, 2)
        self._crs_transformer = IdentityCRSTransformer()
        self.mock._get_chip.return_value = np.random.rand(1, 2, 2, 3)

        if raster is not None:
            self._extent = Box(0, 0, raster.shape[0], raster.shape[1])
            self._dtype = raster.dtype

            def get_chip(window):
                return raster[window.ymin:window.ymax, window.xmin:
                              window.xmax, :]

            self.mock._get_chip.side_effect = get_chip

    @property
    def dtype(self):
        return self._dtype

    @property
    def crs_transformer(self):
        return self._crs_transformer

    def _get_chip(self, window):
        return self.mock._get_chip(window)

    def set_raster(self, raster):
        self.set_return_vals(raster=raster)
