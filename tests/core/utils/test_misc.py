from typing import Literal
import unittest

from shapely.ops import unary_union

from rastervision.core.box import Box
from rastervision.core.utils.misc import (calculate_required_padding,
                                          ensure_tuple)


def windows_cover_extent(extent: Box, windows: list[Box]) -> bool:
    windows_union = unary_union([w.to_shapely() for w in windows])
    return Box.within_aoi(extent, windows_union)


class TestCalculateRequiredPadding(unittest.TestCase):
    def _test_box_get_windows_with_padding(
            self, extent_sz: tuple[int, int], chip_sz: tuple[int, int],
            stride: tuple[int, int],
            pad_direction: Literal['start', 'end', 'both'],
            crop_sz: int | None):
        extent = Box(0, 0, *extent_sz)
        padding = calculate_required_padding(
            extent_sz=extent.size,
            chip_sz=chip_sz,
            stride=stride,
            pad_direction=pad_direction,
            crop_sz=crop_sz,
        )
        windows = extent.get_windows(
            chip_sz, stride, padding=padding, pad_direction=pad_direction)
        self.assertTrue(windows_cover_extent(extent, windows))

    def test_without_crop_sz(self):
        for pad_direction in ['start', 'end', 'both']:
            self._test_box_get_windows_with_padding(
                extent_sz=(100, 100),
                chip_sz=(20, 40),
                stride=(20, 20),
                pad_direction=pad_direction,
                crop_sz=None,
            )
            self._test_box_get_windows_with_padding(
                extent_sz=(100, 100),
                chip_sz=(23, 23),
                stride=(19, 19),
                pad_direction=pad_direction,
                crop_sz=None,
            )
            self._test_box_get_windows_with_padding(
                extent_sz=(10, 10),
                chip_sz=(8, 8),
                stride=(5, 8),
                pad_direction=pad_direction,
                crop_sz=None,
            )
            self._test_box_get_windows_with_padding(
                extent_sz=(10, 10),
                chip_sz=(12, 12),
                stride=(5, 5),
                pad_direction=pad_direction,
                crop_sz=None,
            )

    def test_with_crop_sz(self):
        self._test_box_get_windows_with_padding(
            extent_sz=(100, 100),
            chip_sz=(24, 24),
            stride=(20, 20),
            pad_direction='both',
            crop_sz=2,
        )
        self._test_box_get_windows_with_padding(
            extent_sz=(100, 100),
            chip_sz=(23, 23),
            stride=(11, 13),
            pad_direction='both',
            crop_sz=2,
        )

    def test_error_if_chip_sz_lt_stride(self):
        args = dict(
            extent_sz=(100, 100),
            chip_sz=(10, 10),
            stride=(20, 20),
            pad_direction='both',
        )
        self.assertRaises(ValueError,
                          lambda: calculate_required_padding(**args))

    def test_error_if_cropped_chip_sz_lt_stride(self):
        args = dict(
            extent_sz=(100, 100),
            chip_sz=(20, 20),
            stride=(20, 20),
            pad_direction='both',
            crop_sz=5,
        )
        self.assertRaises(ValueError,
                          lambda: calculate_required_padding(**args))

    def test_error_if_crop_sz_with_wrong_pad_dir(self):
        args = dict(
            extent_sz=(100, 100),
            chip_sz=(50, 50),
            stride=(20, 20),
            pad_direction='end',
            crop_sz=5,
        )
        self.assertRaises(ValueError,
                          lambda: calculate_required_padding(**args))


class TestEnsureTuple(unittest.TestCase):
    def test(self):
        self.assertTupleEqual(ensure_tuple((1, 1), n=2), (1, 1))
        self.assertTupleEqual(ensure_tuple(1, n=2), (1, 1))
        self.assertTupleEqual(ensure_tuple(1, n=4), (1, 1, 1, 1))
        self.assertRaises(ValueError, lambda: ensure_tuple((1, 1, 1), n=2))


if __name__ == '__main__':
    unittest.main()
