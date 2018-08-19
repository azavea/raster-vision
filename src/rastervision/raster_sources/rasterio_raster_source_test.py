import unittest
import tempfile
import os

import rasterio
import numpy as np

from rastervision.raster_sources.rasterio_raster_source import (load_window)
from rastervision.core.box import Box


class RasterioRasterSourceTest(unittest.TestCase):
    def test_load_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # make geotiff filled with ones and zeros with nodata == 1
            image_path = os.path.join(temp_dir, 'temp.tif')
            height = 100
            width = 100
            nb_channels = 3
            image_dataset = rasterio.open(
                image_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=nb_channels,
                dtype=np.uint8,
                nodata=1)
            im = np.random.randint(0, 2, (height, width, nb_channels)).astype(
                np.uint8)
            for channel in range(nb_channels):
                image_dataset.write(im[:, :, channel], channel + 1)

            # Should be all zeros after converting nodata values to zero.
            window = Box.make_square(0, 0, 100).rasterio_format()
            chip = load_window(image_dataset, window=window)
            np.testing.assert_equal(chip, np.zeros(chip.shape))


if __name__ == '__main__':
    unittest.main()
