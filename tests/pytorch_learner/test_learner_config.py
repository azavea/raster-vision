import unittest

from rastervision.pytorch_learner import (GeoDataWindowConfig,
                                          GeoDataWindowMethod)


class TestGeoDataWindowConfig(unittest.TestCase):
    def test_default_stride(self):
        cfg = GeoDataWindowConfig(method=GeoDataWindowMethod.sliding, size=256)
        self.assertEqual(cfg.stride, 256)

    def test_set_stride(self):
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding, size=256, stride=15)
        self.assertEqual(cfg.stride, 15)
