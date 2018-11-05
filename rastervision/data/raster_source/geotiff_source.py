import subprocess
import os
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
    log.info('Building VRT...')
    image_paths = [download_if_needed(uri, temp_dir) for uri in image_uris]
    image_path = os.path.join(temp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


class GeoTiffSource(RasterioRasterSource):
    def __init__(self, uris, raster_transformers, temp_dir,
                 channel_order=None):
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
