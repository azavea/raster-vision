import logging
import math
import os
import pyproj
import subprocess
from decimal import Decimal

from rastervision.core.box import Box
from rastervision.data.crs_transformer import RasterioCRSTransformer
from rastervision.data.raster_source.rasterio_source \
    import RasterioRasterSource
from rastervision.utils.files import download_if_needed

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


class GeoTiffSource(RasterioRasterSource):
    def __init__(self,
                 uris,
                 raster_transformers,
                 temp_dir,
                 channel_order=None,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0):
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters
        self.uris = uris
        super().__init__(raster_transformers, temp_dir, channel_order)

    def _download_data(self, temp_dir):
        if len(self.uris) == 1:
            return download_if_needed(self.uris[0], temp_dir)
        else:
            return download_and_build_vrt(self.uris, temp_dir)

    def _set_crs_transformer(self):
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)

    def _get_chip(self, window):
        no_shift = self.x_shift_meters == 0.0 and self.y_shift_meters == 0.0
        yes_shift = not no_shift
        if yes_shift:
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

        return super()._get_chip(window)

    def _activate(self):
        super()._activate()
        self.crs = self.image_dataset.crs
        if self.crs:
            self.proj = pyproj.Proj(self.crs)
        else:
            self.proj = None
        self.crs = str(self.crs)
