from unittest.mock import Mock
import numpy as np

from rastervision.core import Box
from rastervision.core.data import (RasterSource, IdentityCRSTransformer,
                                    ActivateMixin)


class MockRasterSource(RasterSource, ActivateMixin):
    def __init__(self, channel_order, num_channels, raster_transformers=[]):
        super().__init__(channel_order, num_channels, raster_transformers)
        self.mock = Mock()
        self.set_return_vals()

    def set_return_vals(self, raster=None):
        self.mock.get_extent.return_value = Box.make_square(0, 0, 2)
        self.mock.get_dtype.return_value = np.uint8
        self.mock.get_crs_transformer.return_value = IdentityCRSTransformer()
        self.mock._get_chip.return_value = np.random.rand(1, 2, 2, 3)

        if raster is not None:
            self.mock.get_extent.return_value = Box(0, 0, raster.shape[0],
                                                    raster.shape[1])
            self.mock.get_dtype.return_value = raster.dtype

            def get_chip(window):
                return raster[window.ymin:window.ymax, window.xmin:
                              window.xmax, :]

            self.mock._get_chip.side_effect = get_chip

    def get_extent(self):
        return self.mock.get_extent()

    def get_dtype(self):
        return self.mock.get_dtype()

    def get_crs_transformer(self):
        return self.mock.get_crs_transformer()

    def _get_chip(self, window):
        return self.mock._get_chip(window)

    def set_raster(self, raster):
        self.set_return_vals(raster=raster)

    def _activate(self):
        pass

    def _deactivate(self):
        pass
