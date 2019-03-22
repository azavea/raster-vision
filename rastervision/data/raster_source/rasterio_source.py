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

from rastervision.core.box import Box
from rastervision.data.crs_transformer import RasterioCRSTransformer
from rastervision.utils.files import download_if_needed
from rastervision.data import (ActivateMixin, ActivationError)
from rastervision.data.raster_source import RasterSource

log = logging.getLogger(__name__)
wgs84 = pyproj.Proj({'init': 'epsg:4326'})
wgs84_proj4 = '+init=epsg:4326'
meters_per_degree = 111319.5


def build_vrt(vrt_path, image_paths):
    """Build a VRT for a set of TIFF files."""
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris, temp_dir):
    log.info('Building VRT...')
    image_paths = [download_if_needed(uri, temp_dir) for uri in image_uris]
    image_path = os.path.join(temp_dir, 'index.vrt')
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


class RasterioSource(ActivateMixin, RasterSource):
    def __init__(self,
                 uris,
                 raster_transformers,
                 temp_dir,
                 channel_order=None,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0):
        """Constructor.

        This RasterSource can read any file that can be opened by Rasterio/GDAL
        including georeferenced formats such as GeoTIFF and non-georeferenced formats
        such as JPG. See https://www.gdal.org/formats_list.html for more details.

        If channel_order is None, then use non-alpha channels. This also sets any
        masked or NODATA pixel values to be zeros.

        Args:
            channel_order: list of indices of channels to extract from raw imagery
        """
        self.uris = uris
        self.temp_dir = temp_dir
        self.image_temp_dir = None
        self.image_dataset = None
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters

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

    def _download_data(self, temp_dir):
        """Download any data needed for this Raster Source.

        Return a single local path representing the image or a VRT of the data.
        """
        if len(self.uris) == 1:
            return download_if_needed(self.uris[0], temp_dir)
        else:
            return download_and_build_vrt(self.uris, temp_dir)

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
        shifted_window = self._get_shifted_window(window)
        return load_window(
            self.image_dataset,
            window=shifted_window.rasterio_format(),
            is_masked=self.is_masked)

    def _activate(self):
        # Download images to temporary directory and delete it when done.
        self.image_temp_dir = tempfile.TemporaryDirectory(dir=self.temp_dir)
        self.imagery_path = self._download_data(self.image_temp_dir.name)
        self.image_dataset = rasterio.open(self.imagery_path)
        self._set_crs_transformer()

    def _set_crs_transformer(self):
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)
        self.crs = self.image_dataset.crs
        if self.crs:
            self.proj = pyproj.Proj(self.crs)
        else:
            self.proj = None
        self.crs = str(self.crs)

    def _deactivate(self):
        self.image_dataset.close()
        self.image_dataset = None
        self.image_temp_dir.cleanup()
        self.image_temp_dir = None

    def _get_shifted_window(self, window):
        do_shift = self.x_shift_meters != 0.0 or self.y_shift_meters != 0.0
        if do_shift:
            ymin, xmin, ymax, xmax = window.tuple_format()
            width = window.get_width()
            height = window.get_height()

            # Transform image coordinates into world coordinates
            transform = self.image_dataset.transform
            xmin2, ymin2 = transform * (xmin, ymin)

            # Transform from world coordinates to WGS84
            if self.crs != wgs84_proj4 and self.proj:
                lon, lat = pyproj.transform(self.proj, wgs84, xmin2, ymin2)
            else:
                lon, lat = xmin2, ymin2

            # Shift.  This is performed by computing the shifts in
            # meters to shifts in degrees.  Those shifts are then
            # applied to the WGS84 coordinate.
            #
            # Courtesy of https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters  # noqa
            lat_radians = math.pi * lat / 180.0
            dlon = Decimal(self.x_shift_meters) / Decimal(
                meters_per_degree * math.cos(lat_radians))
            dlat = Decimal(self.y_shift_meters) / Decimal(meters_per_degree)
            lon = float(Decimal(lon) + dlon)
            lat = float(Decimal(lat) + dlat)

            # Transform from WGS84 to world coordinates
            if self.crs != wgs84_proj4 and self.proj:
                xmin3, ymin3 = pyproj.transform(wgs84, self.proj, lon, lat)
                xmin3 = int(round(xmin3))
                ymin3 = int(round(ymin3))
            else:
                xmin3, ymin3 = lon, lat

            # Trasnform from world coordinates back into image coordinates
            xmin4, ymin4 = ~transform * (xmin3, ymin3)

            window = Box(ymin4, xmin4, ymin4 + height, xmin4 + width)
        return window
