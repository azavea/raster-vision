from abc import abstractmethod
import tempfile

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags)

from rastervision.data import (ActivateMixin, ActivationError)
from rastervision.data.raster_source import RasterSource
from rastervision.core.box import Box


def load_window(image_dataset, window=None, is_masked=False):
    """Load a window of an image using Rasterio.

    Args:
        image_dataset: a Rasterio dataset
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
        is_masked: If True, read a masked array from rasterio

    Returns:
        np.ndarray of shape (height, width, channels) where channels is the number of
            channels in the image_dataset.
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

    im = np.transpose(im, axes=[1, 2, 0])
    return im


class RasterioRasterSource(ActivateMixin, RasterSource):
    def __init__(self, raster_transformers, temp_dir, channel_order=None):
        """Constructor.

        If channel_order is None, then use non-alpha channels. This also sets any
        masked or NODATA pixel values to be zeros.

        Args:
            channel_order: list of indices of channels to extract from raw imagery
        """
        self.temp_dir = temp_dir
        self.image_temp_dir = None
        self.image_dataset = None
        num_channels = None

        # Activate in order to get information out of the raster
        with self.activate():
            num_channels = self.image_dataset.count
            if channel_order is None:
                colorinterp = self.image_dataset.colorinterp
                if colorinterp:
                    channel_order = [
                        i for i, color_interp in enumerate(colorinterp)
                        if color_interp != ColorInterp.alpha
                    ]
                else:
                    channel_order = list(range(0, num_channels))
            self.validate_channel_order(channel_order, num_channels)

            mask_flags = self.image_dataset.mask_flag_enums
            self.is_masked = any(
                [m for m in mask_flags if m != MaskFlags.all_valid])

            self.height = self.image_dataset.height
            self.width = self.image_dataset.width

            # Get 1x1 chip and apply raster transformers to test dtype.
            test_chip = self.get_raw_chip(Box.make_square(0, 0, 1))
            test_chip = test_chip[:, :, channel_order]
            for transformer in raster_transformers:
                test_chip = transformer.transform(test_chip, channel_order)
            self.dtype = test_chip.dtype

            self._set_crs_transformer()

        super().__init__(channel_order, num_channels, raster_transformers)

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
        return load_window(
            self.image_dataset,
            window=window.rasterio_format(),
            is_masked=self.is_masked)

    def _activate(self):
        # Download images to temporary directory and delete it when done.
        self.image_temp_dir = tempfile.TemporaryDirectory(dir=self.temp_dir)
        self.imagery_path = self._download_data(self.image_temp_dir.name)
        self.image_dataset = rasterio.open(self.imagery_path)
        self._set_crs_transformer()

    def _deactivate(self):
        self.image_dataset.close()
        self.image_dataset = None
        self.image_temp_dir.cleanup()
        self.image_temp_dir = None
