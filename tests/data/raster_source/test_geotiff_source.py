import unittest
import os

import numpy as np
import rasterio

import rastervision as rv
from rastervision.core import Box
from rastervision.utils.misc import save_img
from rastervision.data.raster_source.rasterio_source import load_window
from rastervision.rv_config import RVConfig

from tests import data_file_path


class TestGeoTiffSource(unittest.TestCase):
    def test_load_window(self):
        with RVConfig.get_tmp_dir() as temp_dir:
            # make geotiff filled with ones and zeros with nodata == 1
            image_path = os.path.join(temp_dir, 'temp.tif')
            height = 100
            width = 100
            nb_channels = 3
            with rasterio.open(
                    image_path,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=nb_channels,
                    dtype=np.uint8,
                    nodata=1) as image_dataset:
                im = np.random.randint(
                    0, 2, (height, width, nb_channels)).astype(np.uint8)
                for channel in range(nb_channels):
                    image_dataset.write(im[:, :, channel], channel + 1)

            # Should be all zeros after converting nodata values to zero.
            window = Box.make_square(0, 0, 100).rasterio_format()
            with rasterio.open(image_path) as image_dataset:
                chip = load_window(image_dataset, window=window)
            np.testing.assert_equal(chip, np.zeros(chip.shape))

    def test_get_dtype(self):
        img_path = data_file_path('small-rgb-tile.tif')
        with RVConfig.get_tmp_dir() as tmp_dir:
            source = rv.data.GeoTiffSourceConfig(uris=[img_path]) \
                            .create_source(tmp_dir)

            self.assertEqual(source.get_dtype(), np.uint8)

    def test_gets_raw_chip(self):
        img_path = data_file_path('small-rgb-tile.tif')
        channel_order = [0, 1]

        source = rv.data.GeoTiffSourceConfig(uris=[img_path],
                                             channel_order=channel_order) \
                        .create_source(tmp_dir=None)

        with source.activate():
            out_chip = source.get_raw_image_array()
            self.assertEqual(out_chip.shape[2], 3)

    def test_gets_raw_chip_from_proto(self):
        img_path = data_file_path('small-rgb-tile.tif')
        channel_order = [0, 1]

        msg = rv.data.GeoTiffSourceConfig(uris=[img_path],
                                          channel_order=channel_order) \
                     .to_proto()

        source = rv.RasterSourceConfig.from_proto(msg) \
                                      .create_source(tmp_dir=None)

        with source.activate():
            out_chip = source.get_raw_image_array()
            self.assertEqual(out_chip.shape[2], 3)

    def test_uses_channel_order(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            img_path = os.path.join(tmp_dir, 'img.tif')
            chip = np.ones((2, 2, 4)).astype(np.uint8)
            chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
            save_img(chip, img_path)

            channel_order = [0, 1, 2]
            source = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                          .with_uri(img_path) \
                                          .with_channel_order(channel_order) \
                                          .build() \
                                          .create_source(tmp_dir=tmp_dir)
            with source.activate():
                out_chip = source.get_image_array()
                expected_out_chip = np.ones((2, 2, 3)).astype(np.uint8)
                expected_out_chip[:, :, :] *= np.array([0, 1,
                                                        2]).astype(np.uint8)
                np.testing.assert_equal(out_chip, expected_out_chip)

    def test_with_stats_transformer(self):
        config = rv.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE) \
                                      .with_uri('dummy') \
                                      .with_stats_transformer() \
                                      .build()

        self.assertEqual(len(config.transformers), 1)
        self.assertIsInstance(config.transformers[0],
                              rv.data.StatsTransformerConfig)

    def test_missing_config_uri(self):
        with self.assertRaises(rv.ConfigError):
            rv.data.RasterSourceConfig.builder(rv.GEOTIFF_SOURCE).build()

    def test_no_missing_config(self):
        try:
            rv.data.RasterSourceConfig.builder(
                rv.GEOTIFF_SOURCE).with_uri('').build()
        except rv.ConfigError:
            self.fail('ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
