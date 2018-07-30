import unittest
import numpy as np

from rastervision.core.box import Box
from rastervision.core.class_map import (ClassMap, ClassItem)
from rastervision.ml_backends.tf_deeplab import (
    numpy_to_png, png_to_numpy, create_tf_example, parse_tf_example)


class TFDeeplabTest(unittest.TestCase):
    def test_png_roundtrip_1channel(self):
        array1 = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
        png = numpy_to_png(array1)
        array2 = png_to_numpy(png)
        self.assertEqual(np.equal(array1, array2).all(), True)

    def test_png_roundtrip_3channels(self):
        array1 = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        png = numpy_to_png(array1)
        array2 = png_to_numpy(png)
        self.assertEqual(np.equal(array1, array2).all(), True)

    def test_tfrecord_roundtrip(self):
        image1 = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
        window = Box(0, 0, 72, 72)
        labels1 = np.random.randint(0, 20, size=(10, 10), dtype=np.uint8)
        class_array = [
            ClassItem(0, 'beachball'),
            ClassItem(1, 'log'),
            ClassItem(2, 'dog'),
            ClassItem(3, 'frog'),
            ClassItem(4, 'noodle'),
            ClassItem(5, 'poodle'),
            ClassItem(6, 'doodle')
        ]
        class_map = ClassMap(class_array)
        example = create_tf_example(image1, window, labels1, class_map)
        flabels1 = (labels1 <= 6) * labels1
        image2, labels2 = parse_tf_example(example)
        images_boolean = np.equal(image1, image2).all()
        labels_boolean = np.equal(flabels1, labels2).all()
        self.assertEqual(images_boolean and labels_boolean, True)


if __name__ == '__main__':
    unittest.main()
