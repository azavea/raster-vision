import unittest

from rastervision.pytorch_learner.dataset.visualizer import SemanticSegmentationVisualizer


class TestVisualizer(unittest.TestCase):
    def test_get_batch_from_empty_dataset(self):
        viz = SemanticSegmentationVisualizer(
            class_names=['background', 'building'],
            class_colors=['lightgray', 'darkred'])
        ds = []

        with self.assertRaises(ValueError):
            viz.get_batch(ds)
