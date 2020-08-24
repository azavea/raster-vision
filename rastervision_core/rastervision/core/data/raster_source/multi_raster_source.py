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
from rastervision.core.data.crs_transformer import CRSTransformer
from rastervision.core.data.utils import all_equal


class MultiRasterSourceError(Exception):
    pass


class MultiRasterSource(ActivateMixin, RasterSource):
    def __init__(self,
                 raster_sources,
                 raw_channel_order,
                 raster_transformers=[]):

        num_channels = len(raw_channel_order)
        channel_order = list(range(num_channels))
        super().__init__(channel_order, num_channels, raster_transformers)

        self.raster_sources = raster_sources
        self.raw_channel_order = raw_channel_order
        self.num_channels = len(self.raw_channel_order)
        self.crs_transformer = CRSTransformer()
        self.validate_raster_sources()

    def validate_raster_sources(self):
        dtypes = [rs.get_dtype() for rs in self.raster_sources]
        if not all_equal(dtypes):
            raise MultiRasterSourceError(
                'dtypes of all sub raster sources must be same')

        num_channels = sum(rs.num_channels for rs in self.raster_sources)
        if num_channels != self.num_channels:
            raise MultiRasterSourceError(
                'num_channels and channel mappings for sub raster sources do '
                'not match')

    def _activate(self):
        for rs in self.raster_sources:
            rs._activate()

    def _deactivate(self):
        for rs in self.raster_sources:
            rs._deactivate()

    def get_extent(self):
        rs = self.raster_sources[0]
        extent = rs.get_extent()
        return extent

    def get_dtype(self):
        rs = self.raster_sources[0]
        dtype = rs.get_dtype()
        return dtype

    def get_crs_transformer(self):
        return self.crs_transformer

    def _get_chip(self, window):
        """Return the raw chip located in the window.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        chip_slices = [rs._get_chip(window) for rs in self.raster_sources]
        chip = np.concatenate(chip_slices, axis=-1)
        chip = chip[..., self.raw_channel_order]
        return chip

    def get_raw_chip(self, window):
        """Return raw chip without using channel_order or applying transforms.

        Args:
            window: (Box) the window for which to get the chip

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        return self._get_chip(window)

    def get_raw_image_array(self):
        """Return entire raw image without using channel_order or applying transforms.

        Not safe to call on very large RasterSources.
        """
        window = self.get_extent()
        return self.get_raw_chip(window)

    def get_image_array(self):
        """Return entire transformed image array.

        Not safe to call on very large RasterSources.
        """
        window = self.get_extent()
        return self.get_chip(window)
