import unittest
from os.path import join
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio as rio
from torchvision.datasets.folder import DatasetFolder

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.data.utils import write_window
from rastervision.pytorch_learner.dataset import (discover_images, load_image,
                                                  make_image_folder_dataset)
from rastervision.pytorch_backend.pytorch_learner_backend import write_chip


class TestUtils(unittest.TestCase):
    def test_discover_images(self):
        with get_tmp_dir() as tmp_dir:
            chip = np.random.randint(
                0, 256, size=(100, 100, 3), dtype=np.uint8)
            path_1 = join(tmp_dir, 'test.png')
            write_chip(chip, path_1)

            chip = np.random.randint(
                0, 256, size=(100, 100, 8), dtype=np.uint8)
            path_2 = join(tmp_dir, 'test.npy')
            write_chip(chip, path_2)

            with open(join(tmp_dir, 'test.txt'), 'w') as f:
                f.write('abc')

            paths = discover_images(tmp_dir)
            self.assertEqual(len(paths), 2)
            self.assertIn(Path(path_1), paths)
            self.assertIn(Path(path_2), paths)

    def test_load_image(self):
        with get_tmp_dir() as tmp_dir:
            chip = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
            path = join(tmp_dir, '1.png')
            write_chip(chip, path)
            np.testing.assert_array_equal(
                load_image(path), chip[..., np.newaxis])

            chip = np.random.randint(
                0, 256, size=(100, 100, 3), dtype=np.uint8)
            path = join(tmp_dir, '2.png')
            write_chip(chip, path)
            np.testing.assert_array_equal(load_image(path), chip)

            chip = np.random.randint(
                0, 256, size=(100, 100, 8), dtype=np.uint8)
            path = join(tmp_dir, '3.npy')
            write_chip(chip, path)
            np.testing.assert_array_equal(load_image(path), chip)

            chip = np.random.randint(
                0, 256, size=(100, 100, 8), dtype=np.uint8)
            path = join(tmp_dir, '4.tif')
            profile = dict(height=100, width=100, count=8, dtype=np.uint8)
            with rio.open(path, 'w', **profile) as ds:
                write_window(ds, chip)
            np.testing.assert_array_equal(load_image(path), chip)

    def test_make_image_folder_dataset(self):
        with get_tmp_dir() as tmp_dir:
            with TemporaryDirectory(dir=tmp_dir) as dir_a, TemporaryDirectory(
                    dir=tmp_dir) as dir_b:
                chip = np.random.randint(
                    0, 256, size=(100, 100, 3), dtype=np.uint8)
                path_1 = join(dir_a, 'test.png')
                write_chip(chip, path_1)

                chip = np.random.randint(
                    0, 256, size=(100, 100, 8), dtype=np.uint8)
                path_2 = join(dir_b, 'test.npy')
                write_chip(chip, path_2)

                ds = make_image_folder_dataset(tmp_dir)
                self.assertIsInstance(ds, DatasetFolder)
                self.assertEqual(len(ds), 2)

                ds = make_image_folder_dataset(tmp_dir, classes=[dir_a])
                self.assertIsInstance(ds, DatasetFolder)
                self.assertEqual(len(ds), 1)


if __name__ == '__main__':
    unittest.main()
