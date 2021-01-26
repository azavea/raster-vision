import logging
import math
import os
from pyproj import Transformer
import subprocess
from decimal import Decimal
import tempfile
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags)

from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import (RasterSource, CropOffsets)
from rastervision.core.data import (ActivateMixin, ActivationError)

log = logging.getLogger(__name__)
wgs84 = 'epsg:4326'
meters_per_degree = 111319.5


def build_vrt(vrt_path, image_paths):
    """Build a VRT for a set of TIFF files."""
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris, tmp_dir):
    log.info('Building VRT...')
    image_paths = [download_if_needed(uri, tmp_dir) for uri in image_uris]
    image_path = os.path.join(tmp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


def stream_and_build_vrt(images_uris, tmp_dir):
    log.info('Building VRT...')
    image_paths = images_uris
    image_path = os.path.join(tmp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


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


def fill_overflow(extent: Box,
                  window: Box,
                  arr: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Given a window and corresponding array of values, if the window
    overflows the extent, fill the overflowing regions with fill_value.
    """
    top_overflow = max(0, extent.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - extent.ymax)
    left_overflow = max(0, extent.xmin - window.xmin)
    right_overflow = max(0, window.xmax - extent.xmax)

    h, w = arr.shape[:2]
    arr[:top_overflow] = fill_value
    arr[h - bottom_overflow:] = fill_value
    arr[:, :left_overflow] = fill_value
    arr[:, w - right_overflow:] = fill_value
    return arr


class RasterioSource(ActivateMixin, RasterSource):
    def __init__(self,
                 uris,
                 raster_transformers,
                 tmp_dir,
                 allow_streaming=False,
                 channel_order=None,
                 x_shift=0.0,
                 y_shift=0.0,
                 extent_crop: Optional[CropOffsets] = None):
        """Constructor.

        This RasterSource can read any file that can be opened by Rasterio/GDAL
        including georeferenced formats such as GeoTIFF and non-georeferenced formats
        such as JPG. See https://www.gdal.org/formats_list.html for more details.

        If channel_order is None, then use non-alpha channels. This also sets any
        masked or NODATA pixel values to be zeros.

        Args:
            channel_order: list of indices of channels to extract from raw imagery
            extent_crop (CropOffsets, optional): Relative
                offsets (top, left, bottom, right) for cropping the extent.
                Useful for using splitting a scene into different datasets.
                Defaults to None i.e. no cropping.
        """
        self.uris = uris
        self.tmp_dir = tmp_dir
        self.image_tmp_dir = None
        self.image_dataset = None
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.do_shift = self.x_shift != 0.0 or self.y_shift != 0.0
        self.allow_streaming = allow_streaming
        self.extent_crop = extent_crop

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

    def _download_data(self, tmp_dir):
        """Download any data needed for this Raster Source.

        Return a single local path representing the image or a VRT of the data.
        """
        if len(self.uris) == 1:
            if self.allow_streaming:
                return self.uris[0]
            else:
                return download_if_needed(self.uris[0], tmp_dir)
        else:
            if self.allow_streaming:
                return stream_and_build_vrt(self.uris, tmp_dir)
            else:
                return download_and_build_vrt(self.uris, tmp_dir)

    def get_crs_transformer(self):
        return self.crs_transformer

    def get_extent(self):
        h, w = self.height, self.width
        if self.extent_crop is not None:
            skip_top, skip_left, skip_bottom, skip_right = self.extent_crop
            ymin, xmin = int(h * skip_top), int(w * skip_left)
            ymax, xmax = h - int(h * skip_bottom), w - int(w * skip_right)
            return Box(ymin, xmin, ymax, xmax)
        return Box(0, 0, h, w)

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        return self.dtype

    def _get_chip(self, window: Box) -> np.ndarray:
        if self.image_dataset is None:
            raise ActivationError('RasterSource must be activated before use')
        shifted_window = self._get_shifted_window(window)
        chip = load_window(
            self.image_dataset,
            window=shifted_window.rasterio_format(),
            is_masked=self.is_masked)
        if self.extent_crop is not None:
            chip = fill_overflow(self.get_extent(), window, chip)
        return chip

    def _activate(self):
        # Download images to temporary directory and delete them when done.
        self.image_tmp_dir = tempfile.TemporaryDirectory(dir=self.tmp_dir)
        self.imagery_path = self._download_data(self.image_tmp_dir.name)
        self.image_dataset = rasterio.open(self.imagery_path)
        self._set_crs_transformer()

    def _set_crs_transformer(self):
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)
        crs = self.image_dataset.crs
        self.to_wgs84 = None
        self.from_wgs84 = None
        if crs and self.do_shift:
            self.to_wgs84 = Transformer.from_crs(
                crs.wkt, wgs84, always_xy=True)
            self.from_wgs84 = Transformer.from_crs(
                wgs84, crs.wkt, always_xy=True)

    def _deactivate(self):
        self.image_dataset.close()
        self.image_dataset = None
        self.image_tmp_dir.cleanup()
        self.image_tmp_dir = None

    def _get_shifted_window(self, window):
        do_shift = self.x_shift != 0.0 or self.y_shift != 0.0
        if do_shift:
            ymin, xmin, ymax, xmax = window.tuple_format()
            width = window.get_width()
            height = window.get_height()

            # Transform image coordinates into world coordinates
            transform = self.image_dataset.transform
            xmin2, ymin2 = transform * (xmin, ymin)

            # Transform from world coordinates to WGS84
            if self.to_wgs84:
                lon, lat = self.to_wgs84.transform(xmin2, ymin2)
            else:
                lon, lat = xmin2, ymin2

            # Shift.  This is performed by computing the shifts in
            # meters to shifts in degrees.  Those shifts are then
            # applied to the WGS84 coordinate.
            #
            # Courtesy of https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters  # noqa
            lat_radians = math.pi * lat / 180.0
            dlon = Decimal(self.x_shift) / Decimal(
                meters_per_degree * math.cos(lat_radians))
            dlat = Decimal(self.y_shift) / Decimal(meters_per_degree)
            lon = float(Decimal(lon) + dlon)
            lat = float(Decimal(lat) + dlat)

            # Transform from WGS84 to world coordinates
            if self.from_wgs84:
                xmin3, ymin3 = self.from_wgs84.transform(lon, lat)
            else:
                xmin3, ymin3 = lon, lat

            # Trasnform from world coordinates back into image coordinates
            xmin4, ymin4 = ~transform * (xmin3, ymin3)

            window = Box(ymin4, xmin4, ymin4 + height, xmin4 + width)
        return window
