import unittest

import numpy as np

from rastervision.core.box import Box, BoxSizeError
from shapely.geometry import box as ShapelyBox

np.random.seed(1)


class TestBox(unittest.TestCase):
    def setUp(self):
        self.ymin = 0
        self.xmin = 0
        self.ymax = 2
        self.xmax = 3
        self.box = Box(self.ymin, self.xmin, self.ymax, self.xmax)

    def test_get_height(self):
        height = self.ymax - self.ymin
        self.assertEqual(self.box.get_height(), height)

    def test_get_width(self):
        width = self.xmax - self.xmin
        self.assertEqual(self.box.get_width(), width)

    def test_get_area(self):
        area = self.box.get_height() * self.box.get_width()
        self.assertEqual(self.box.get_area(), area)

    def test_rasterio_format(self):
        rasterio_box = ((self.ymin, self.ymax), (self.xmin, self.xmax))
        self.assertEqual(self.box.rasterio_format(), rasterio_box)

    def test_tuple_format(self):
        box_tuple = (0, 0, 2, 3)
        output_box = self.box.tuple_format()
        self.assertEqual(output_box, box_tuple)

    def test_shapely_format(self):
        shapely_box = (self.xmin, self.ymin, self.xmax, self.ymax)
        self.assertEqual(self.box.shapely_format(), shapely_box)

    def test_npbox_format(self):
        self.assertEqual(
            tuple(self.box.npbox_format()), self.box.tuple_format())
        self.assertEqual(self.box.npbox_format().dtype, np.float)

    def test_geojson_coordinates(self):
        nw = (self.xmin, self.ymin)
        ne = (self.xmin, self.ymax)
        se = (self.xmax, self.ymax)
        sw = (self.xmax, self.ymin)
        geojson_coords = [nw, ne, se, sw, nw]
        self.assertEqual(self.box.geojson_coordinates(), geojson_coords)

    def test_make_random_square_container(self):
        size = 5
        nb_tests = 100
        for _ in range(nb_tests):
            container = self.box.make_random_square_container(size)
            self.assertEqual(container.get_width(), container.get_height())
            self.assertEqual(container.get_width(), size)
            self.assertTrue(container.get_shapely().contains(
                self.box.get_shapely()))

    def test_make_random_square_container_too_big(self):
        size = 1
        with self.assertRaises(BoxSizeError):
            self.box.make_random_square_container(size)

    def test_make_random_square(self):
        window = Box(5, 5, 15, 15)
        size = 5
        nb_tests = 100
        for _ in range(nb_tests):
            box = window.make_random_square(size)
            self.assertEqual(box.get_width(), box.get_height())
            self.assertEqual(box.get_width(), size)
            self.assertTrue(window.get_shapely().contains(box.get_shapely()))

    def test_from_npbox(self):
        npbox = np.array([self.ymin, self.xmin, self.ymax, self.xmax])
        output_box = Box.from_npbox(npbox)
        self.assertEqual(output_box, self.box)

    def test_from_shapely(self):
        shape = ShapelyBox(self.xmin, self.ymin, self.xmax, self.ymax)
        output_box = Box.from_shapely(shape)
        self.assertEqual(output_box, self.box)

    def test_get_shapely(self):
        bounds = self.box.get_shapely().bounds
        self.assertEqual((bounds[1], bounds[0], bounds[3], bounds[2]),
                         self.box.tuple_format())

    def test_make_square(self):
        square = Box(0, 0, 10, 10)
        output_square = Box.make_square(0, 0, 10)
        self.assertEqual(output_square, square)
        self.assertEqual(output_square.get_width(), output_square.get_height())

    def test_make_eroded(self):
        max_extent = Box.make_square(0, 0, 10)
        box = Box(1, 1, 3, 4)
        buffer_size = erosion_size = 1
        eroded_box = box.make_buffer(buffer_size, max_extent) \
                        .make_eroded(erosion_size)
        self.assertEqual(eroded_box, box)

    def test_make_buffer(self):
        buffer_size = 1
        max_extent = Box.make_square(0, 0, 3)
        buffer_box = Box(0, 0, 3, 3)
        output_buffer_box = self.box.make_buffer(buffer_size, max_extent)
        self.assertEqual(output_buffer_box, buffer_box)

        buffer_size = 0.5
        max_extent = Box.make_square(0, 0, 5)
        buffer_box = Box(0, 0, 3, 5)
        output_buffer_box = self.box.make_buffer(buffer_size, max_extent)
        self.assertEqual(output_buffer_box, buffer_box)

    def test_make_copy(self):
        copy_box = self.box.make_copy()
        self.assertIsNot(copy_box, self.box)
        self.assertEqual(copy_box, self.box)

    def test_get_windows(self):
        extent = Box(0, 0, 100, 100)
        windows = list(extent.get_windows(10, 10))
        self.assertEqual(len(windows), 100)

        extent = Box(0, 0, 100, 100)
        windows = list(extent.get_windows(10, 5))
        self.assertEqual(len(windows), 400)

        extent = Box(0, 0, 20, 20)
        windows = set(
            [window.tuple_format() for window in extent.get_windows(10, 10)])
        expected_windows = [
            Box.make_square(0, 0, 10),
            Box.make_square(10, 0, 10),
            Box.make_square(0, 10, 10),
            Box.make_square(10, 10, 10)
        ]
        expected_windows = set(
            [window.tuple_format() for window in expected_windows])
        self.assertSetEqual(windows, expected_windows)


if __name__ == "__main__":
    unittest.main()
