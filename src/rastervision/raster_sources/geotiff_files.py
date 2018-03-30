import subprocess
import os
import rasterio

from rastervision.raster_sources.rasterio_raster_source import (
    RasterioRasterSource)
from rastervision.crs_transformers.rasterio_crs_transformer import (
    RasterioCRSTransformer)
from rastervision.utils.files import download_if_needed


def build_vrt(vrt_path, image_paths):
    """Build a VRT for a set of TIFF files."""
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris, temp_dir):
    print('Downloading and building VRT...')
    image_paths = [download_if_needed(uri, temp_dir) for uri in image_uris]
    image_path = os.path.join(temp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


class GeoTiffFiles(RasterioRasterSource):
    def __init__(self, raster_transformer, uris):
        self.uris = uris
        super().__init__(raster_transformer)

    def build_image_dataset(self):
        print('Loading GeoTiffFFiles...')
        imagery_path = download_and_build_vrt(
            self.uris, self.temp_dir.name)
        return rasterio.open(imagery_path)

    def get_crs_transformer(self):
        return RasterioCRSTransformer(self.image_dataset)
