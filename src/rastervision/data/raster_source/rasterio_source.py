import numpy as np

from rastervision.data.raster_source import RasterSource
from rastervision.core.box import Box


def load_window(image_dataset, window=None):
    """Load a window of an image from a TIFF file.

    Args:
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
    """
    im = image_dataset.read(window=window, boundless=True)

    # Handle non-zero NODATA values by setting the data to 0.
    for channel, nodata in enumerate(image_dataset.nodatavals):
        if nodata is not None and nodata != 0:
            im[channel, im[channel] == nodata] = 0

    im = np.transpose(im, axes=[1, 2, 0])
    return im


class RasterioRasterSource(RasterSource):
    def __init__(self, raster_transformers, temp_dir, channel_order=None):
        self.temp_dir = temp_dir
        self.image_dataset = self.build_image_dataset(temp_dir)
        super().__init__(raster_transformers, channel_order)

    def build_image_dataset(self, temp_dir):
        pass

    def get_extent(self):
        return Box(0, 0, self.image_dataset.height, self.image_dataset.width)

    def _get_chip(self, window):
        return load_window(self.image_dataset, window.rasterio_format())
