import unittest
import os
from tempfile import TemporaryDirectory

import numpy as np

import rastervision as rv
from rastervision.core.raster_stats import RasterStats
from rastervision.utils.misc import save_img


class TestImageSource(unittest.TestCase):
    def test_applies_transforms(self):
        with TemporaryDirectory() as tmp_dir:
            stats_uri = os.path.join(tmp_dir, "stats.json")
            img_path = os.path.join(tmp_dir, "img.tif")

            raster_stats = RasterStats()
            raster_stats.means = np.ones((4, ))
            raster_stats.stds = np.ones((4, )) * 2
            raster_stats.save(stats_uri)

            channel_order = [0, 1, 2]

            stats_transformer = rv.RasterTransformerConfig.builder(rv.STATS_TRANSFORMER) \
                                                          .with_stats_uri(stats_uri) \
                                                          .build()

            transformers = [stats_transformer]

            chip = (np.ones((2, 2, 4)) * [3, 3, 3, 0]).astype(np.uint16)
            save_img(chip, img_path)

            source = rv.RasterSourceConfig.builder(rv.IMAGE_SOURCE) \
                                          .with_uri(img_path) \
                                          .with_channel_order(channel_order) \
                                          .with_transformers(transformers) \
                                          .build() \
                                          .create_source(tmp_dir)

            out_chip = source.get_image_array()
            expected_out_chip = np.ones((2, 2, 3)) * 170
            np.testing.assert_equal(out_chip, expected_out_chip)
