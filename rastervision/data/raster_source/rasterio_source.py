from abc import abstractmethod

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags)

from rastervision.data import (ActivateMixin, ActivationError)
from rastervision.data.raster_source import RasterSource
from rastervision.core.box import Box


def load_window(image_dataset, window=None, channels=None, is_masked=False):
    """Load a window of an image from a TIFF file.

    Args:
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
        channels: An optional list of bands to read.
        is_masked: If True, read a  masked array from rasterio
    """
    if is_masked:
        im = image_dataset.read(window=window, boundless=True, masked=True)
        im = np.ma.filled(im, fill_value=0)
    else:
        im = image_dataset.read(window=window, boundless=True)

    # Handle non-zero NODATA values by setting the data to 0.
    for channel, nodata in enumerate(image_dataset.nodatavals):
        if nodata is not None and nodata != 0:
            im[channel, im[channel] == nodata] = 0

    if channels:
        im = im[channels, :]
    im = np.transpose(im, axes=[1, 2, 0])
    return im


class RasterioRasterSource(ActivateMixin, RasterSource):
    def __init__(self, raster_transformers, temp_dir, channel_order=None):
        super().__init__(channel_order, raster_transformers)

        self.temp_dir = temp_dir
        self.imagery_path = self._download_data(temp_dir)

        # Activate in order to get information out of the raster
        with self.activate():
            colorinterp = self.image_dataset.colorinterp
            self.channels = [
                i for i, color_interp in enumerate(colorinterp)
                if color_interp != ColorInterp.alpha
            ]

            mask_flags = self.image_dataset.mask_flag_enums
            self.is_masked = any(
                [m for m in mask_flags if m != MaskFlags.all_valid])

            self.height = self.image_dataset.height
            self.width = self.image_dataset.width
            # Get 1x1 chip (after applying raster transformers) to test dtype
            # and channel order if needed
            test_chip = self.get_chip(Box.make_square(0, 0, 1))
            self.dtype = test_chip.dtype

            self.channel_order = self.channel_order or list(
                range(0, test_chip.shape[2]))

            self._set_crs_transformer()

    @abstractmethod
    def _download_data(self, tmp_dir):
        """Download any data needed for this Raster Source.
        Return a single local path representing the image or a VRT of the data."""
        pass

    def get_crs_transformer(self):
        return self.crs_transformer

    def get_extent(self):
        return Box(0, 0, self.height, self.width)

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        return self.dtype

    def _get_chip(self, window):
        if self.image_dataset is None:
            raise ActivationError('RasterSource must be activated before use')
        return load_window(self.image_dataset, window.rasterio_format(),
                           self.channels)

    def _activate(self):
        self.image_dataset = rasterio.open(self.imagery_path)

    def _deactivate(self):
        self.image_dataset.close()
        self.image_dataset = None
