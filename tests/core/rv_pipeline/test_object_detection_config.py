import unittest

from rastervision.core.rv_pipeline import (ObjectDetectionPredictOptions)


class TestObjectDetectionPredictOptions(unittest.TestCase):
    def test_stride_validator(self):
        cfg = ObjectDetectionPredictOptions(chip_sz=10)
        self.assertEqual(cfg.stride, 5)
        cfg = ObjectDetectionPredictOptions(chip_sz=11)
        self.assertEqual(cfg.stride, 5)


if __name__ == '__main__':
    unittest.main()
