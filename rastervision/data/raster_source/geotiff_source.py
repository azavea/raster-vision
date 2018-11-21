import subprocess
import os
import logging
from decimal import Decimal

from rastervision.data.raster_source.rasterio_source \
    import RasterioRasterSource
from rastervision.data.crs_transformer import RasterioCRSTransformer
from rastervision.utils.files import download_if_needed

log = logging.getLogger(__name__)


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


def modify_geotransform_for_shift(vrt_path, dx, dy):
    import xml
    import pyproj
    import rasterio as rio
    import math

    tree = xml.etree.ElementTree.parse(vrt_path)
    geotransform_elt = tree.getroot().find('GeoTransform')
    a, b, c, d, e, f = list(map(float, geotransform_elt.text.split(',')))

    # Project the upper-left corner coordinates in EPSG 4326
    with rio.open(vrt_path, 'r') as ds:
        crs = ds.crs
    if str(crs) != '+init=epsg:4326':
        vrtproj = pyproj.Proj(crs)
        wgs84proj = pyproj.Proj({'init': 'epsg:4326'})
        lon, lat = pyproj.transform(vrtproj, wgs84proj, a, d)
    else:
        lon, lat = a, d

    # Calculate the offsets
    latr = math.pi * lat / 180.0
    dx = Decimal(dx) / Decimal(111319.5 * math.cos(latr))
    dy = Decimal(dy) / Decimal(111319.5)

    # Compute the shifted coordinates
    lon = float(Decimal(lon) + dx)
    lat = float(Decimal(lat) + dy)

    # Put the upper-left corner coordinates back into the original CRStransformer
    if str(crs) != '+init=epsg:4326':
        new_a, new_d = pyproj.transform(wgs84proj, vrtproj, lon, lat)
    else:
        new_a, new_d = lon, lat

    # Edit the VRT
    geotransform_text = ' {:.16e},  {:.16e},  {:.16e},  {:.16e},  {:.16e},  {:.16e}'.format(  # noqa
        new_a, b, c, new_d, e, f)
    geotransform_elt.text = geotransform_text
    tree.write(vrt_path)


class GeoTiffSource(RasterioRasterSource):
    def __init__(self,
                 uris,
                 raster_transformers,
                 temp_dir,
                 channel_order=None,
                 x_shift_meters=0.0,
                 y_shift_meters=0.0):
        self.uris = uris
        self.x_shift_meters = x_shift_meters
        self.y_shift_meters = y_shift_meters
        super().__init__(raster_transformers, temp_dir, channel_order)

    def _download_data(self, temp_dir):
        no_shift = self.x_shift_meters == 0.0 and self.y_shift_meters == 0.0

        if len(self.uris) == 1 and no_shift:
            return download_if_needed(self.uris[0], temp_dir)
        else:
            vrt_path = download_and_build_vrt(self.uris, temp_dir)
            if not no_shift:
                dx = self.x_shift_meters
                dy = self.y_shift_meters
                modify_geotransform_for_shift(vrt_path, dx, dy)
            return vrt_path

    def _set_crs_transformer(self):
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)
