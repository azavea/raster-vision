import unittest
import os
from tempfile import TemporaryDirectory

import numpy as np

import rastervision as rv
from rastervision.core.raster_stats import RasterStats
from rastervision.utils.misc import save_img

from tests import data_file_path

# TODO: Test Proto methods

class TestGeoTiffSource(unittest.TestCase):
    def test_gets_raw_chip(self):
        img_path = data_file_path("small-rgb-tile.tif")
        channel_order = [0, 1]

        source = rv.data.GeoTiffSourceConfig(uris=[img_path], channel_order=[0,1]) \
                        .create_source(tmp_dir=None)

        out_chip = source.get_raw_image_array()
        self.assertEqual(out_chip.shape[2], 3)

    def test_uses_channel_order(self):
        with TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "img.tif")
            chip = np.ones((2, 2, 4)).astype(np.uint8)
            chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
            save_img(chip, img_path)

            channel_order = [0, 1, 2]
            source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                          .with_uri(img_path) \
                                          .with_channel_order(channel_order) \
                                          .build() \
                                          .create_source(tmp_dir=None)

        out_chip = source.get_image_array()
        expected_out_chip = np.ones((2, 2, 3)).astype(np.uint8)
        expected_out_chip[:, :, :] *= np.array([0, 1, 2]).astype(np.uint8)
        np.testing.assert_equal(out_chip, expected_out_chip)

    def test_with_stats_transformer(self):
        config = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                      .with_uri("dummy") \
                                      .with_stats_transformer() \
                                      .build()

        self.assertEqual(len(config.transformers), 1)
        self.assertIsInstance(config.transformers[0], rv.data.StatsTransformerConfig)
