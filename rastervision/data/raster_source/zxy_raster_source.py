import logging
import os
import pyproj
import uuid
from os.path import join

import numpy as np
import rasterio

from rastervision.data.crs_transformer import RasterioCRSTransformer
from rastervision.data.raster_source.rasterio_source import (RasterioSource,
                                                             RasterSource)
from rastervision.utils.zxy2geotiff import _zxy2geotiff
from rastervision.utils.files import make_dir

log = logging.getLogger(__name__)


class ZXYRasterSource(RasterioSource):
    def __init__(self,
                 tile_schema,
                 zoom,
                 bounds,
                 raster_transformers,
                 temp_dir,
                 channel_order=None,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0):
        """Construct a raster source that can read from a z/x/y tile server.

        Args:
            tile_schema: (str) the URI schema for zxy tiles (ie. a slippy map tile server)
                of the form /tileserver-uri/{z}/{x}/{y}.png. If {-y} is used, the tiles
                are assumed to be indexed using TMS coordinates, where the y axis starts
                at the southernmost point. The URI can be for http, S3, or the local
                file system.
            zoom: (int) the zoom level to use when retrieving tiles
            bounds: (list) a list of length 4 containing min_lat, min_lng,
                max_lat, max_lng
            raster_transformers: list of RasterTransformers to apply
            temp_dir: (str) where to store temporary files
            channel_order: list of indices of channels to extract from raw
                imagery
            x_shift_meters: (float) A number of meters to shift along the
                x-axis. A ositive shift moves the "camera" to the right.
            y_shift_meters: A number of meters to shift along the y-axis. A
                positive shift moves the "camera" down.
        """
        self.tile_schema = tile_schema
        self.zoom = zoom
        self.bounds = bounds
        self.image_dataset = None
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters

        self.temp_dir = temp_dir
        self.geotiff_path = join(temp_dir, str(uuid.uuid4()), 'image.geotiff')
        make_dir(self.geotiff_path, use_dirname=True)
        height, width, transform = _zxy2geotiff(
            self.tile_schema, zoom, bounds, self.geotiff_path, dry_run=True)

        self.height = height
        self.width = width
        self.dtype = np.uint8
        self.transform = transform
        self.is_masked = False
        self._set_crs_transformer()

        num_channels = 3
        RasterSource.__init__(self, channel_order, num_channels,
                              raster_transformers)

    def _activate(self):
        _zxy2geotiff(self.tile_schema, self.zoom, self.bounds,
                     self.geotiff_path)
        self.image_dataset = rasterio.open(self.geotiff_path)
        self._set_crs_transformer()

    def _set_crs_transformer(self):
        self.crs = 'epsg:3857'
        self.crs_transformer = RasterioCRSTransformer(self.transform, self.crs)
        self.proj = pyproj.Proj(init=self.crs)

    def _deactivate(self):
        self.image_dataset.close()
        self.image_dataset = None
        os.remove(self.geotiff_path)
