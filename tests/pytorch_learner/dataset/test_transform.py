import unittest

import numpy as np
import albumentations as A

from rastervision.pytorch_learner.dataset.transform import (
    yxyx_to_albu, albu_to_yxyx, xywh_to_albu, apply_transform)


class TestTransforms(unittest.TestCase):
    def test_apply_transform_invalid_ndims(self):
        tf = A.Resize(10, 10)
        with self.assertRaises(NotImplementedError):
            x = np.ones((1, 2, 5, 5, 3))  # 5 dims
            apply_transform(tf, image=x)

        with self.assertRaises(NotImplementedError):
            x = np.ones((5, 5))  # 2 dims
            apply_transform(tf, image=x)

    def test_apply_transform_3d(self):
        tf = A.Resize(10, 10)
        x = np.ones((5, 5, 3))
        y = np.ones((5, 5), dtype=int)
        x_tf = apply_transform(tf, image=x, mask=y)
        self.assertEqual(x_tf['image'].shape, (10, 10, 3))
        self.assertEqual(x_tf['mask'].shape, (10, 10))

    def test_apply_transform_4d(self):
        tf = A.Resize(10, 10)
        x = np.ones((2, 5, 5, 3))
        y = np.ones((5, 5), dtype=int)
        x_tf = apply_transform(tf, image=x, mask=y)
        self.assertEqual(x_tf['image'].shape, (2, 10, 10, 3))
        self.assertEqual(x_tf['mask'].shape, (10, 10))

    def test_box_format_conversions_yxyx(self):
        boxes = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 10],
            ], dtype=float)
        boxes_albu_gt = np.array(
            [
                [.2, .1, .4, .3],
                [.3, .2, 1., .4],
            ], dtype=float)
        boxes_albu = yxyx_to_albu(boxes, (10, 10))
        self.assertTrue(np.all(boxes_albu == boxes_albu_gt))

        boxes_yxyx = albu_to_yxyx(boxes_albu, (10, 10))
        self.assertTrue(np.all(boxes_yxyx == boxes))

    def test_box_format_conversions_xywh(self):
        boxes = np.array(
            [
                [1, 2, 3, 4],
                [2, 3, 4, 10],
            ], dtype=float)
        boxes_albu_gt = np.array(
            [
                [.1, .2, .4, .6],
                [.2, .3, .6, 1.],
            ], dtype=float)
        boxes_albu = xywh_to_albu(boxes, (10, 10))
        np.testing.assert_allclose(boxes_albu, boxes_albu_gt)


if __name__ == '__main__':
    unittest.main()
