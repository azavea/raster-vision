import logging
import math
import os
import pyproj
import subprocess
from decimal import Decimal
import tempfile

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags)

from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data import (ActivateMixin, ActivationError)


class MultiRasterSource(ActivateMixin, RasterSource):

	def __init__(self, raster_sources, raster_transformers=[]):
		self.raster_sources = raster_sources
        self.raster_transformers = raster_transformers

    def _activate(self):
        for rs in self.raster_sources:
			rs._activate()

    def _deactivate(self):
        for rs in self.raster_sources:
			rs._deactivate()

    def get_extent(self):
        raise NotImplementedError()

    def get_dtype(self):
        raise NotImplementedError()

    def get_crs_transformer(self):
        raise NotImplementedError()

    def _get_chip(self, window):
        """Return the raw chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        chip_slices = [rs._get_chip(window) for rs in self.raster_sources]
		chip = np.concatenate(chip_slices, axis=-1)
		return chip

    def get_raw_chip(self, window):
        """Return raw chip without using channel_order or applying transforms.

        Args:
            window: (Box) the window for which to get the chip

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        return self._get_chip(window)

    def get_image_array(self):
        """Return entire transformed image array.

        Not safe to call on very large RasterSources.
        """
        raise NotImplementedError()

    def get_raw_image_array(self):
        """Return entire raw image without using channel_order or applying transforms.

        Not safe to call on very large RasterSources.
        """
        raise NotImplementedError()
