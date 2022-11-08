import unittest

import numpy as np
from shapely.geometry import box as ShapelyBox

from rastervision.core.box import Box, BoxSizeError, RioWindow

np.random.seed(1)


class TestBox(unittest.TestCase):
    def setUp(self):
        self.ymin = 0
        self.xmin = 0
        self.ymax = 2
        self.xmax = 3
        self.box = Box(self.ymin, self.xmin, self.ymax, self.xmax)

    def test_reproject(self):
        def transform(point):
            (y, x) = point
            return ((y + 1) // 2, x // 2)

        reproj = self.box.reproject(transform)
        self.assertTrue(reproj.xmin == 0)
        self.assertTrue(reproj.ymin == 0)
        self.assertTrue(reproj.ymax == 1)
        self.assertTrue(reproj.xmax == 2)

    def test_dict(self):
        dictionary = self.box.to_dict()
        other = Box.from_dict(dictionary)
        self.assertTrue(self.box == other)

    def test_bad_square(self):
        self.assertRaises(BoxSizeError,
                          lambda: self.box.make_random_square(10))

    def test_bad_conatiner(self):
        self.assertRaises(BoxSizeError,
                          lambda: self.box.make_random_square_container(1))

    def test_neq(self):
        other = Box(self.ymin + 1, self.xmin, self.ymax, self.xmax)
        self.assertTrue(self.box != other)

    def test_int(self):
        other = Box(
            float(self.ymin) + 0.01, float(self.xmin), float(self.ymax),
            float(self.xmax))
        self.assertTrue(other.to_int() == self.box)

    def test_height(self):
        height = self.ymax - self.ymin
        self.assertEqual(self.box.height, height)

    def test_width(self):
        width = self.xmax - self.xmin
        self.assertEqual(self.box.width, width)

    def test_area(self):
        area = self.box.height * self.box.width
        self.assertEqual(self.box.area, area)

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
        self.assertEqual(self.box.npbox_format().dtype, float)

    def test_geojson_coordinates(self):
        nw = [self.xmin, self.ymin]
        ne = [self.xmin, self.ymax]
        se = [self.xmax, self.ymax]
        sw = [self.xmax, self.ymin]
        geojson_coords = [nw, ne, se, sw, nw]
        self.assertEqual(self.box.geojson_coordinates(), geojson_coords)

    def test_make_random_square_container(self):
        size = 5
        nb_tests = 100
        for _ in range(nb_tests):
            container = self.box.make_random_square_container(size)
            self.assertEqual(container.width, container.height)
            self.assertEqual(container.width, size)
            self.assertTrue(container.to_shapely().contains(
                self.box.to_shapely()))

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
            self.assertEqual(box.width, box.height)
            self.assertEqual(box.width, size)
            self.assertTrue(window.to_shapely().contains(box.to_shapely()))

    def test_from_npbox(self):
        npbox = np.array([self.ymin, self.xmin, self.ymax, self.xmax])
        output_box = Box.from_npbox(npbox)
        self.assertEqual(output_box, self.box)

    def test_from_shapely(self):
        shape = ShapelyBox(self.xmin, self.ymin, self.xmax, self.ymax)
        output_box = Box.from_shapely(shape)
        self.assertEqual(output_box, self.box)

    def test_to_xywh(self):
        x, y, w, h = np.random.randint(100, size=4)
        box = Box(y, x, y + h, x + w)
        self.assertEqual(box.to_xywh(), (x, y, w, h))

    def test_to_xyxy(self):
        box = Box(*np.random.randint(100, size=4))
        self.assertEqual(box.to_xyxy(),
                         (box.xmin, box.ymin, box.xmax, box.ymax))

    def test_to_points(self):
        box = Box(0, 0, 1, 2)
        points = box.to_points()
        points = set(tuple(p) for p in points.tolist())
        self.assertIn((0, 0), points)
        self.assertIn((2, 0), points)
        self.assertIn((2, 1), points)
        self.assertIn((0, 1), points)

    def test_to_slices(self):
        box = Box(*np.random.randint(100, size=4))
        hslice, wslice = box.to_slices()
        self.assertEqual(hslice, slice(box.ymin, box.ymax))
        self.assertEqual(wslice, slice(box.xmin, box.xmax))

    def test_to_rasterio(self):
        x, y, w, h = np.random.randint(100, size=4)
        box = Box(y, x, y + h, x + w)
        rio_w = box.to_rasterio()
        self.assertEqual(
            (rio_w.col_off, rio_w.row_off, rio_w.width, rio_w.height),
            (x, y, w, h))

    def test_from_rasterio(self):
        rio_w = RioWindow.from_slices((10, 20), (1, 2))
        box = Box.from_rasterio(rio_w)
        self.assertEqual(box, Box(10, 1, 20, 2))

    def test_intersects(self):
        box = Box(0, 0, 10, 10)
        self.assertTrue(box.intersects(Box(0, 5, 10, 15)))
        self.assertTrue(box.intersects(Box(-5, 0, 5, 10)))
        self.assertFalse(box.intersects(Box(20, 20, 30, 30)))
        self.assertFalse(box.intersects(Box(0, 10, 10, 20)))
        self.assertFalse(box.intersects(Box(10, 0, 20, 10)))

    def test_translate(self):
        box = Box(*np.random.randint(100, size=4))
        dy, dx = np.random.randint(100, size=2)
        self.assertEqual(
            box.translate(dy, dx),
            Box(box.ymin + dy, box.xmin + dx, box.ymax + dy, box.xmax + dx))

    def test_shift_origin(self):
        extent = Box(5, 5, 8, 8)
        box = Box(0, 0, 10, 10)
        self.assertEqual(box.shift_origin(extent), Box(5, 5, 15, 15))

    def test_to_offsets(self):
        outer = Box(10, 10, 20, 20)
        inner = Box(15, 15, 18, 18)
        self.assertEqual(inner.to_offsets(outer), Box(5, 5, 8, 8))

    def test_to_shapely(self):
        bounds = self.box.to_shapely().bounds
        self.assertEqual((bounds[1], bounds[0], bounds[3], bounds[2]),
                         self.box.tuple_format())

    def test_make_square(self):
        square = Box(0, 0, 10, 10)
        output_square = Box.make_square(0, 0, 10)
        self.assertEqual(output_square, square)
        self.assertEqual(output_square.width, output_square.height)

    def test_erode(self):
        max_extent = Box.make_square(0, 0, 10)
        box = Box(1, 1, 3, 4)
        buffer_size = erosion_size = 1
        eroded_box = box.buffer(buffer_size, max_extent).erode(erosion_size)
        self.assertEqual(eroded_box, box)

    def test_center_crop(self):
        box_in = Box(0, 0, 10, 10)
        box_out = box_in.center_crop(2, 4)
        self.assertEqual(box_out, Box(2, 4, 8, 6))

    def test_buffer(self):
        buffer_size = 1
        max_extent = Box.make_square(0, 0, 3)
        buffer_box = Box(0, 0, 3, 3)
        output_buffer_box = self.box.buffer(buffer_size, max_extent)
        self.assertEqual(output_buffer_box, buffer_box)

        buffer_size = 0.5
        max_extent = Box.make_square(0, 0, 5)
        buffer_box = Box(0, 0, 3, 5)
        output_buffer_box = self.box.buffer(buffer_size, max_extent)
        self.assertEqual(output_buffer_box, buffer_box)

    def test_copy(self):
        copy_box = self.box.copy()
        self.assertIsNot(copy_box, self.box)
        self.assertEqual(copy_box, self.box)

    def test_pad(self):
        box_in = Box(10, 10, 20, 20)
        box_out = box_in.pad(ymin=20, xmin=20, ymax=20, xmax=20)
        self.assertEqual(box_out, Box(-10, -10, 40, 40))

        box_in = Box(10, 10, 20, 20)
        box_out = box_in.pad(ymin=20, xmin=0, ymax=0, xmax=20)
        self.assertEqual(box_out, Box(-10, 10, 20, 40))

    def test_get_windows(self):
        box = Box(0, 0, 10, 10)
        self.assertRaises(ValueError, lambda: box.get_windows(-1, 1))
        self.assertRaises(ValueError, lambda: box.get_windows(1, -1))
        self.assertRaises(ValueError, lambda: box.get_windows(0, 1))
        self.assertRaises(ValueError, lambda: box.get_windows(1, 0))
        self.assertRaises(ValueError,
                          lambda: box.get_windows(1, 1, padding=-1))

        extent = Box(0, 0, 100, 100)
        windows = extent.get_windows(size=10, stride=10)
        self.assertEqual(len(windows), 10 * 10)

        extent = Box(0, 0, 100, 100)
        windows = extent.get_windows(size=10, stride=5)
        self.assertEqual(len(windows), 20 * 20)

        extent = Box(0, 0, 20, 20)
        windows = set(extent.get_windows(size=10, stride=10))
        expected_windows = set([
            Box.make_square(0, 0, 10),
            Box.make_square(0, 10, 10),
            Box.make_square(10, 0, 10),
            Box.make_square(10, 10, 10)
        ])
        self.assertSetEqual(windows, expected_windows)

        extent = Box(10, 10, 20, 20)
        windows = set(extent.get_windows(size=6, stride=6))
        expected_windows = set([
            Box.make_square(10, 10, 6),
            Box.make_square(10, 16, 6),
            Box.make_square(16, 10, 6),
            Box.make_square(16, 16, 6)
        ])
        self.assertSetEqual(windows, expected_windows)

        extent = Box(0, 0, 10, 10)
        args = dict(size=5, stride=3, padding=1, pad_direction='end')
        windows = set(extent.get_windows(**args))
        expected_windows = set([
            Box(0, 0, 5, 5),
            Box(0, 3, 5, 8),
            Box(0, 6, 5, 11),
            Box(3, 0, 8, 5),
            Box(3, 3, 8, 8),
            Box(3, 6, 8, 11),
            Box(6, 0, 11, 5),
            Box(6, 3, 11, 8),
            Box(6, 6, 11, 11),
        ])
        arg_str = ', '.join(f'{k}={v!r}' for k, v in args.items())
        msg = f'{extent!r}.get_windows({arg_str})'
        self.assertSetEqual(windows, expected_windows, msg=msg)

        extent = Box(0, 0, 10, 10)
        args = dict(size=5, stride=3, padding=1, pad_direction='start')
        windows = set(extent.get_windows(**args))
        expected_windows = set([
            Box(-1, -1, 4, 4),
            Box(-1, 2, 4, 7),
            Box(-1, 5, 4, 10),
            Box(2, -1, 7, 4),
            Box(2, 2, 7, 7),
            Box(2, 5, 7, 10),
            Box(5, -1, 10, 4),
            Box(5, 2, 10, 7),
            Box(5, 5, 10, 10),
        ])
        arg_str = ', '.join(f'{k}={v!r}' for k, v in args.items())
        msg = f'{extent!r}.get_windows({arg_str})'
        self.assertSetEqual(windows, expected_windows, msg=msg)

        extent = Box(0, 0, 10, 10)
        args = dict(size=5, stride=3, padding=1, pad_direction='both')
        windows = set(extent.get_windows(**args))
        expected_windows = set([
            Box(-1, -1, 4, 4),
            Box(-1, 2, 4, 7),
            Box(-1, 5, 4, 10),
            Box(2, -1, 7, 4),
            Box(2, 2, 7, 7),
            Box(2, 5, 7, 10),
            Box(5, -1, 10, 4),
            Box(5, 2, 10, 7),
            Box(5, 5, 10, 10),
        ])
        arg_str = ', '.join(f'{k}={v!r}' for k, v in args.items())
        msg = f'{extent!r}.get_windows({arg_str})'
        self.assertSetEqual(windows, expected_windows, msg=msg)

        extent = Box(0, 0, 10, 10)
        args = dict(size=5, stride=3, padding=2, pad_direction='both')
        windows = set(extent.get_windows(**args))
        expected_windows = set([
            Box(-2, -2, 3, 3),
            Box(-2, 1, 3, 6),
            Box(-2, 4, 3, 9),
            Box(-2, 7, 3, 12),
            Box(1, -2, 6, 3),
            Box(1, 1, 6, 6),
            Box(1, 4, 6, 9),
            Box(1, 7, 6, 12),
            Box(4, -2, 9, 3),
            Box(4, 1, 9, 6),
            Box(4, 4, 9, 9),
            Box(4, 7, 9, 12),
            Box(7, -2, 12, 3),
            Box(7, 1, 12, 6),
            Box(7, 4, 12, 9),
            Box(7, 7, 12, 12),
        ])
        arg_str = ', '.join(f'{k}={v!r}' for k, v in args.items())
        msg = f'{extent!r}.get_windows({arg_str})'
        self.assertSetEqual(windows, expected_windows, msg=msg)

    def test_unpacking(self):
        box = Box(1, 2, 3, 4)
        ymin, xmin, ymax, xmax = box
        self.assertEqual((ymin, xmin, ymax, xmax), box.tuple_format())

    def test_subscripting(self):
        box = Box(1, 2, 3, 4)
        self.assertEqual(box[0], 1)
        self.assertEqual(box[1], 2)
        self.assertEqual(box[2], 3)
        self.assertEqual(box[3], 4)
        self.assertRaises(IndexError, lambda: box[4])

    def test_repr(self):
        box = Box(1, 2, 3, 4)
        self.assertEqual(box.__repr__(), 'Box(ymin=1, xmin=2, ymax=3, xmax=4)')

    def test_filter_by_aoi(self):
        windows = [Box.make_square(0, 0, 2), Box.make_square(0, 2, 2)]
        aoi_polygons = [Box.make_square(0, 0, 3).to_shapely()]

        filt_windows = Box.filter_by_aoi(windows, aoi_polygons, within=False)
        self.assertListEqual(filt_windows, windows)

        filt_windows = Box.filter_by_aoi(windows, aoi_polygons, within=True)
        self.assertListEqual(filt_windows, windows[0:1])


if __name__ == '__main__':
    unittest.main()
