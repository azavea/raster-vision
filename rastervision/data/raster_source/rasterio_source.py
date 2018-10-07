from abc import abstractmethod

import numpy as np
from rasterio.enums import (ColorInterp, MaskFlags)

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


class RasterioRasterSource(RasterSource):
    def __init__(self, raster_transformers, temp_dir, channel_order=None):
        self.temp_dir = temp_dir
        self.image_dataset = self.build_image_dataset(temp_dir)
        super().__init__(raster_transformers, channel_order)

        colorinterp = self.image_dataset.colorinterp
        self.channels = [
            i for i, color_interp in enumerate(colorinterp)
            if color_interp != ColorInterp.alpha
        ]

        mask_flags = self.image_dataset.mask_flag_enums
        self.is_masked = any(
            [m for m in mask_flags if m != MaskFlags.all_valid])

    @abstractmethod
    def build_image_dataset(self, temp_dir):
        pass

    def get_extent(self):
        return Box(0, 0, self.image_dataset.height, self.image_dataset.width)

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        # Get 1x1 chip (after applying raster transformers) to test dtype.
        chip = self.get_chip(window=Box.make_square(0, 0, 1))
        return chip.dtype

    def _get_chip(self, window):
        return load_window(self.image_dataset, window.rasterio_format(),
                           self.channels)
