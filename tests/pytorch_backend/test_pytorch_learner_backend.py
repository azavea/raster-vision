import unittest
from os.path import join

import numpy as np
from PIL import Image

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.pytorch_backend.pytorch_learner_backend import (
    get_image_ext, write_chip)


class TestUtils(unittest.TestCase):
    def test_get_image_ext(self):
        chip = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        self.assertEqual(get_image_ext(chip), 'png')
        chip = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        self.assertEqual(get_image_ext(chip), 'png')
        chip = np.random.randint(0, 256, size=(100, 100, 1), dtype=np.uint8)
        self.assertEqual(get_image_ext(chip), 'npy')
        chip = np.random.randint(0, 256, size=(100, 100, 2), dtype=np.uint8)
        self.assertEqual(get_image_ext(chip), 'npy')
        chip = np.random.randint(0, 256, size=(100, 100, 8), dtype=np.uint8)
        self.assertEqual(get_image_ext(chip), 'npy')

    def test_write_chip(self):
        with get_tmp_dir() as tmp_dir:
            chip = np.random.randint(0, 256, size=(100, 100, 3))
            path = join(tmp_dir, 'test.png')
            write_chip(chip, path)
            np.testing.assert_array_equal(np.array(Image.open(path)), chip)

            chip = np.random.randint(0, 256, size=(100, 100, 8))
            path = join(tmp_dir, 'test.npy')
            write_chip(chip, path)
            np.testing.assert_array_equal(np.load(path), chip)


if __name__ == '__main__':
    unittest.main()
