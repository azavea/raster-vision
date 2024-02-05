from typing import Callable
import unittest

from pydantic import ValidationError
import numpy as np

from rastervision.core.data import (ClassConfig, DatasetConfig)
from rastervision.pipeline.config import build_config
from rastervision.core.rv_pipeline.semantic_segmentation_config import (
    SemanticSegmentationConfig, SemanticSegmentationPredictOptions,
    SemanticSegmentationChipOptions, ss_config_upgrader)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (SemanticSegmentationModelConfig,
                                          SolverConfig,
                                          SemanticSegmentationImageDataConfig)


class TestSemanticSegmentationConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_upgrader(self):
        class_config = ClassConfig(
            names=['red', 'green'], colors=['red', 'green'])
        dataset_cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[],
            validation_scenes=[],
            test_scenes=[])
        backend_cfg = PyTorchSemanticSegmentationConfig(
            data=SemanticSegmentationImageDataConfig(),
            model=SemanticSegmentationModelConfig(),
            solver=SolverConfig())
        cfg = SemanticSegmentationConfig(
            dataset=dataset_cfg, backend=backend_cfg)
        old_cfg_dict = cfg.dict()
        old_cfg_dict['channel_display_groups'] = None
        old_cfg_dict['img_format'] = 'npy'
        old_cfg_dict['label_format'] = 'npy'
        new_cfg_dict = ss_config_upgrader(old_cfg_dict, version=0)
        self.assertNotIn('channel_display_groups', new_cfg_dict)
        self.assertNotIn('img_format', new_cfg_dict)
        self.assertNotIn('label_format', new_cfg_dict)
        self.assertNoError(lambda: build_config(new_cfg_dict))


class TestSemanticSegmentationPredictOptions(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_upgrader(self):
        args = dict(stride=None, crop_sz=None)
        self.assertNoError(lambda: SemanticSegmentationPredictOptions(**args))

        args = dict(stride=None, crop_sz='auto')
        self.assertRaises(ValidationError,
                          lambda: SemanticSegmentationPredictOptions(**args))

        args = dict(stride=None, crop_sz=10)
        self.assertRaises(ValidationError,
                          lambda: SemanticSegmentationPredictOptions(**args))

        args = dict(stride=10, crop_sz=0)
        self.assertRaises(ValidationError,
                          lambda: SemanticSegmentationPredictOptions(**args))


class TestSemanticSegmentationChipOptions(unittest.TestCase):
    def test_enough_target_pixels_true(self):
        chip = np.zeros((10, 10, 1), dtype=np.uint8)
        chip[4:, 4:, :] = 1
        chip_options = SemanticSegmentationChipOptions(
            sampling={}, target_class_ids=[1], target_count_threshold=30)
        self.assertTrue(chip_options.enough_target_pixels(chip))

    def test_enough_target_pixels_false(self):
        chip = np.zeros((10, 10, 1), dtype=np.uint8)
        chip[7:, 7:, :] = 1
        chip_options = SemanticSegmentationChipOptions(
            sampling={}, target_class_ids=[1], target_count_threshold=30)
        self.assertFalse(chip_options.enough_target_pixels(chip))


if __name__ == '__main__':
    unittest.main()
