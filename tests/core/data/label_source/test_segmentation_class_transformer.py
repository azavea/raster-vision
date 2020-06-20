import unittest

import numpy as np

from rastervision.core.data import SegmentationClassTransformer
from rastervision.core.data.utils import color_to_triple
from rastervision.core.data.class_config import ClassConfig


class TestSegmentationClassTransformer(unittest.TestCase):
    def setUp(self):
        self.class_config = ClassConfig(
            names=['a', 'b', 'c'], colors=['red', 'green', 'blue'])
        self.class_config.ensure_null_class()
        self.transformer = SegmentationClassTransformer(self.class_config)

        self.rgb_image = np.zeros((1, 3, 3))
        self.rgb_image[0, 0, :] = color_to_triple('red')
        self.rgb_image[0, 1, :] = color_to_triple('green')
        self.rgb_image[0, 2, :] = color_to_triple('blue')

        self.class_image = np.array([[0, 1, 2]])

    def test_rgb_to_class(self):
        class_image = self.transformer.rgb_to_class(self.rgb_image)
        expected_class_image = self.class_image
        np.testing.assert_array_equal(class_image, expected_class_image)

    def test_class_to_rgb(self):
        rgb_image = self.transformer.class_to_rgb(self.class_image)
        expected_rgb_image = self.rgb_image
        np.testing.assert_array_equal(rgb_image, expected_rgb_image)


if __name__ == '__main__':
    unittest.main()
