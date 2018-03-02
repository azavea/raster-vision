import subprocess
import os
import tempfile

import numpy as np
import rasterio

from rv2.core.raster_source import RasterSource
from rv2.core.box import Box
from rv2.core.crs_transformer import CRSTransformer
from rv2.utils.files import download_if_needed, RV_TEMP_DIR


def load_window(image_dataset, window=None):
    """Load a window of an image from a TIFF file.

    Args:
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
    """
    im = np.transpose(
        image_dataset.read(window=window), axes=[1, 2, 0])
    return im


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


class GeoTiffFiles(RasterSource):
    def __init__(self, uris):
        print('Loading GeoTiffFFiles...')
        self.uris = uris
        self.temp_dir = tempfile.TemporaryDirectory(dir=RV_TEMP_DIR)
        self.vrt_path = download_and_build_vrt(
            self.uris, self.temp_dir.name)
        self.image_dataset = rasterio.open(self.vrt_path)

    def get_crs_transformer(self):
        return CRSTransformer(self.image_dataset)

    def get_extent(self):
        return Box(
            0, 0, self.image_dataset.height, self.image_dataset.width)

    def get_chip(self, window):
        return load_window(self.image_dataset, window.rasterio_format())
