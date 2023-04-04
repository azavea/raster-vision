import unittest

import numpy as np

from rastervision.core.data import (BuildingVectorOutputConfig, ClassConfig,
                                    PolygonVectorOutputConfig,
                                    VectorOutputConfig)


class TestVectorOutputConfig(unittest.TestCase):
    def test_get_uri(self):
        # w/o ClasConfig
        cfg = VectorOutputConfig(class_id=0)
        self.assertEqual(cfg.get_uri('abc/def'), 'abc/def/class-0.json')

        # w/ ClasConfig
        class_config = ClassConfig(names=['a', 'b'])
        cfg = VectorOutputConfig(class_id=0)
        self.assertEqual(
            cfg.get_uri('abc/def', class_config), 'abc/def/class-0-a.json')
        cfg = VectorOutputConfig(class_id=1)
        self.assertEqual(
            cfg.get_uri('abc/def', class_config), 'abc/def/class-1-b.json')


class TestPolygonVectorOutputConfig(unittest.TestCase):
    def test_denoise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:65, 60:65] = 1

        # denoise = 0
        cfg = PolygonVectorOutputConfig(class_id=0, denoise=0)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 2)

        # denoise = 8
        cfg = PolygonVectorOutputConfig(class_id=0, denoise=8)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 1)


class TestBuildingVectorOutputConfig(unittest.TestCase):
    def test_denoise(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[60:65, 60:65] = 1

        # denoise = 0
        cfg = BuildingVectorOutputConfig(class_id=0, denoise=0)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 2)

        # denoise = 8
        cfg = BuildingVectorOutputConfig(class_id=0, denoise=8)
        polys = list(cfg.vectorize(mask))
        self.assertEqual(len(polys), 1)


if __name__ == '__main__':
    unittest.main()
