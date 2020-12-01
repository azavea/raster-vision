import unittest

import numpy as np

from rastervision.pytorch_learner.dataset.transform import (
    yxyx_to_albu, albu_to_yxyx, xywh_to_albu)


class TestTransforms(unittest.TestCase):
    def test_box_format_conversions_yxyx(self):
        boxes = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 10],
            ], dtype=np.float32)
        boxes_albu_gt = np.array(
            [
                [.2, .1, .4, .3],
                [.3, .2, 1., .4],
            ], dtype=np.float32)
        boxes_albu = yxyx_to_albu(boxes, (10, 10))
        self.assertTrue(np.all(boxes_albu == boxes_albu_gt))

        boxes_yxyx = albu_to_yxyx(boxes_albu, (10, 10))
        self.assertTrue(np.all(boxes_yxyx == boxes))

    def test_box_format_conversions_xywh(self):
        boxes = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 10],
            ], dtype=np.float32)
        boxes_albu_gt = np.array(
            [
                [.1, .2, .4, .6],
                [.2, .3, .6, 1.],
            ], dtype=np.float32)
        boxes_albu = xywh_to_albu(boxes, (10, 10))
        self.assertTrue(np.all(boxes_albu == boxes_albu_gt))


if __name__ == '__main__':
    unittest.main()
