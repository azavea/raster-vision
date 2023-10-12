from typing import Callable
import unittest

import torch

from rastervision.pytorch_learner.dataset import SemanticSegmentationVisualizer


class TestClassificationVisualizer(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_plot_batch(self):
        # w/o z
        viz = SemanticSegmentationVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 4, 256, 256))
        y = (torch.randn(size=(2, 256, 256)) > 0).long()
        self.assertNoError(lambda: viz.plot_batch(x, y))

        # w/ z
        viz = SemanticSegmentationVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        num_classes = 2
        x = torch.randn(size=(2, 4, 256, 256))
        y = (torch.randn(size=(2, 256, 256)) > 0).long()
        z = torch.randn(size=(2, num_classes, 256, 256)).softmax(dim=-3)
        self.assertNoError(lambda: viz.plot_batch(x, y, z=z))

    def test_plot_batch_temporal(self):
        # w/o z
        viz = SemanticSegmentationVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        x = torch.randn(size=(2, 3, 4, 256, 256))
        y = (torch.randn(size=(2, 256, 256)) > 0).long()
        self.assertNoError(lambda: viz.plot_batch(x, y))
        # w/o z, batch size = 1
        self.assertNoError(lambda: viz.plot_batch(x[[0]], y[[0]]))

        # w/ z
        viz = SemanticSegmentationVisualizer(
            class_names=['bg', 'fg'],
            channel_display_groups=dict(RGB=[0, 1, 2], IR=[3]))
        num_classes = 2
        x = torch.randn(size=(2, 3, 4, 256, 256))
        y = (torch.randn(size=(2, 256, 256)) > 0).long()
        z = torch.randn(size=(2, num_classes, 256, 256)).softmax(dim=-3)
        self.assertNoError(lambda: viz.plot_batch(x, y, z=z))
        # w/ z, batch size = 1
        self.assertNoError(lambda: viz.plot_batch(x[[0]], y[[0]]))
