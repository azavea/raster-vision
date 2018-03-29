import subprocess
import os
import tempfile

import numpy as np
import rasterio

from rastervision.core.raster_source import RasterSource
from rastervision.core.box import Box
from rastervision.crs_transformers.rasterio_crs_transformer import (
    RasterioCRSTransformer)
from rastervision.utils.files import download_if_needed, RV_TEMP_DIR


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
    def __init__(self, raster_transformer, uris):
        print('Loading GeoTiffFFiles...')
        self.uris = uris
        self.temp_dir = tempfile.TemporaryDirectory(dir=RV_TEMP_DIR)
        self.vrt_path = download_and_build_vrt(
            self.uris, self.temp_dir.name)
        self.image_dataset = rasterio.open(self.vrt_path)

        super().__init__(raster_transformer)

    def get_crs_transformer(self):
        return RasterioCRSTransformer(self.image_dataset)

    def get_extent(self):
        return Box(
            0, 0, self.image_dataset.height, self.image_dataset.width)

    def _get_chip(self, window):
        height = window.get_height()
        width = window.get_width()
        # If window is off the edge of the array, the returned image will
        # have a shortened height and width. Therefore, we need to transform
        # the partial chip back to full window size.
        partial_chip = load_window(
            self.image_dataset, window.rasterio_format())
        chip = np.zeros((height, width, partial_chip.shape[2]), dtype=np.uint8)
        chip[0:partial_chip.shape[0], 0:partial_chip.shape[1], :] = \
            partial_chip

        return chip
