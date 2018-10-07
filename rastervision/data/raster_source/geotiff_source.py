import subprocess
import os
import rasterio
import logging

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
    log.info('Downloading and building VRT...')
    image_paths = [download_if_needed(uri, temp_dir) for uri in image_uris]
    image_path = os.path.join(temp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


class GeoTiffSource(RasterioRasterSource):
    def __init__(self, uris, raster_transformers, temp_dir,
                 channel_order=None):
        self.uris = uris
        super().__init__(raster_transformers, temp_dir, channel_order)

    def build_image_dataset(self, temp_dir):
        log.info('Loading GeoTiff files...')
        if len(self.uris) == 1:
            imagery_path = download_if_needed(self.uris[0], temp_dir)
        else:
            imagery_path = download_and_build_vrt(self.uris, temp_dir)
        return rasterio.open(imagery_path)

    def get_crs_transformer(self):
        return RasterioCRSTransformer(self.image_dataset)
