import unittest

import torch

from rastervision.pytorch_learner.dataset.visualizer import SemanticSegmentationVisualizer


class TestVisualizer(unittest.TestCase):
    def test_get_batch_from_empty_dataset(self):
        viz = SemanticSegmentationVisualizer(
            class_names=['background', 'building'],
            class_colors=['lightgray', 'darkred'])
        ds = []

        with self.assertRaises(ValueError):
            viz.get_batch(ds)

    def test_plot_batch_invalid_x_shape(self):
        viz = SemanticSegmentationVisualizer(class_names=['bg', 'fg'])

        y = (torch.randn(size=(2, 256, 256)) > 0).long()

        x = torch.randn(size=(2, 1, 3, 4, 256, 256))
        with self.assertRaises(ValueError):
            viz.plot_batch(x, y)

        x = torch.randn(size=(4, 256, 256))
        with self.assertRaises(ValueError):
            viz.plot_batch(x, y)
