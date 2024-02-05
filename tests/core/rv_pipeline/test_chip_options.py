import unittest

import numpy as np

from rastervision.core.rv_pipeline import (ChipOptions, WindowSamplingConfig,
                                           WindowSamplingMethod)


class TestWindowSamplingConfig(unittest.TestCase):
    def test_default_stride(self):
        cfg = WindowSamplingConfig(
            method=WindowSamplingMethod.sliding, size=256)
        self.assertEqual(cfg.stride, 256)

    def test_set_stride(self):
        cfg = WindowSamplingConfig(
            method=WindowSamplingMethod.sliding, size=256, stride=15)
        self.assertEqual(cfg.stride, 15)


class TestChipOptions(unittest.TestCase):
    def test_keep_chip(self):
        chip_options = ChipOptions(sampling={}, nodata_threshold=0.5)
        chip = np.ones((10, 10, 1), dtype=np.uint8)
        # 49% NODATA
        chip[:7, :7] = 0
        self.assertTrue(chip_options.keep_chip(chip, None))
        # 64% NODATA
        chip[:8, :8] = 0
        self.assertFalse(chip_options.keep_chip(chip, None))

    def test_get_chip_sz(self):
        chip_options = ChipOptions(
            sampling=WindowSamplingConfig(
                method=WindowSamplingMethod.sliding, size=256),
            nodata_threshold=0.5)
        self.assertEqual(chip_options.get_chip_sz(), 256)

        chip_options = ChipOptions(
            sampling=WindowSamplingConfig(
                method=WindowSamplingMethod.random, size=256),
            nodata_threshold=0.5)
        self.assertEqual(chip_options.get_chip_sz(), 256)

        chip_options = ChipOptions(
            sampling=dict(
                scene_1=WindowSamplingConfig(
                    method=WindowSamplingMethod.random, size=256)),
            nodata_threshold=0.5)
        self.assertRaises(KeyError, lambda: chip_options.get_chip_sz())
        self.assertEqual(chip_options.get_chip_sz('scene_1'), 256)
