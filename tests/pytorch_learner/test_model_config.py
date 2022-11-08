from typing import Callable
import unittest

import torch
from torch import nn

from rastervision.pipeline.config import ValidationError
from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.pytorch_learner import (
    Backbone, ExternalModuleConfig, SemanticSegmentationModelConfig,
    ClassificationModelConfig, ObjectDetectionModelConfig)


class TestExternalModuleConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_repo_str_validation(self):
        args = dict(github_repo='abc', entrypoint='foo')
        self.assertRaises(ValidationError,
                          lambda: ExternalModuleConfig(**args))
        args = dict(github_repo='abc/def', entrypoint='foo')
        self.assertNoError(lambda: ExternalModuleConfig(**args))
        args = dict(github_repo='abc/def:xyz', entrypoint='foo')
        self.assertNoError(lambda: ExternalModuleConfig(**args))

    def test_disallow_both_uri_and_repo(self):
        args = dict(uri='abc/def', github_repo='abc/def', entrypoint='foo')
        self.assertRaises(ValidationError,
                          lambda: ExternalModuleConfig(**args))

    def test_build(self):
        with get_tmp_dir() as tmp_dir:
            cfg = ExternalModuleConfig(
                github_repo='AdeelH/pytorch-multi-class-focal-loss:1.1',
                entrypoint='focal_loss',
                entrypoint_kwargs=dict(alpha=[.75, .25], gamma=2))
            loss = cfg.build(tmp_dir)
            self.assertIsInstance(loss, nn.Module)
            self.assertEqual(loss.alpha.tolist(), [.75, .25])
            self.assertEqual(loss.gamma, 2)
            del loss


class TestSemanticSegmentationModelConfig(unittest.TestCase):
    def test_backbone_validation(self):
        args = dict(backboe=Backbone.resnet18)
        self.assertRaises(ValidationError,
                          lambda: SemanticSegmentationModelConfig(**args))

    def test_build(self):
        cfg = SemanticSegmentationModelConfig(pretrained=False)
        model = cfg.build(num_classes=2, in_channels=3)
        model.eval()
        with torch.inference_mode():
            out = model(torch.empty(1, 3, 100, 100))
        self.assertEqual(out['out'].shape, (1, 2, 100, 100))


class TestClassificationModelConfig(unittest.TestCase):
    def test_build(self):
        cfg = ClassificationModelConfig(pretrained=False)
        model = cfg.build(num_classes=2, in_channels=3)
        model.eval()
        with torch.inference_mode():
            out = model(torch.empty(1, 3, 100, 100))
        self.assertEqual(out.shape, (1, 2))


class TestObjectDetectionModelConfig(unittest.TestCase):
    def test_backbone_validation(self):
        args = dict(backboe=Backbone.vgg11)
        self.assertRaises(ValidationError,
                          lambda: ObjectDetectionModelConfig(**args))

    def test_build(self):
        cfg = ObjectDetectionModelConfig(pretrained=False)
        model = cfg.build(num_classes=2, in_channels=3, img_sz=200)
        model.eval()
        with torch.inference_mode():
            out = model(torch.empty(1, 3, 100, 100))
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)
        self.assertIn('boxes', out[0].keys())
        self.assertIn('labels', out[0].keys())
        self.assertIn('scores', out[0].keys())


if __name__ == '__main__':
    unittest.main()
