from typing import Callable
import unittest

import torch

from rastervision.core.box import Box
from rastervision.pytorch_learner.dataset import (ObjectDetectionVisualizer,
                                                  BoxList)


def random_boxlist(x, nboxes: int = 5) -> BoxList:
    extent = Box(0, 0, *x.shape[-2:])
    boxes = [extent.make_random_square(50) for _ in range(nboxes)]
    npboxes = torch.from_numpy(Box.to_npboxes(boxes))
    class_ids = torch.randint(0, 2, size=(nboxes, ))
    scores = torch.rand(nboxes)
    return BoxList(npboxes, class_ids=class_ids, scores=scores)


class TestClassificationVisualizer(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_plot_batch(self):
        # w/o z
        viz = ObjectDetectionVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 4, 256, 256))
        y = [random_boxlist(_x) for _x in x]
        self.assertNoError(lambda: viz.plot_batch(x, y))

        # w/ z
        viz = ObjectDetectionVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 4, 256, 256))
        y = [random_boxlist(_x) for _x in x]
        z = [random_boxlist(_x) for _x in x]
        self.assertNoError(lambda: viz.plot_batch(x, y, z=z))

    def test_plot_batch_temporal(self):
        # w/o z
        viz = ObjectDetectionVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 3, 4, 256, 256))
        y = [random_boxlist(_x) for _x in x]
        self.assertNoError(lambda: viz.plot_batch(x, y))

        # w/ z
        viz = ObjectDetectionVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 3, 4, 256, 256))
        y = [random_boxlist(_x) for _x in x]
        z = [random_boxlist(_x) for _x in x]
        self.assertNoError(lambda: viz.plot_batch(x, y, z=z))
