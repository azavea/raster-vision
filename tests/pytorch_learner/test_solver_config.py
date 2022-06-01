from typing import Callable
import unittest

from torch import nn

from rastervision.pytorch_learner import (SolverConfig, solver_config_upgrader,
                                          ExternalModuleConfig)
from rastervision.pipeline.config import ValidationError, build_config


class TestSolverConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_upgrader(self):
        old_cfg = SolverConfig()

        # ignore_last_class = False
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['ignore_last_class'] = False
        new_cfg_dict = solver_config_upgrader(old_cfg_dict, version=3)
        self.assertNotIn('ignore_last_class', new_cfg_dict)
        self.assertIn('ignore_class_index', new_cfg_dict)
        self.assertIsNone(new_cfg_dict['ignore_class_index'])

        new_cfg = build_config(new_cfg_dict)
        loss = new_cfg.build_loss(num_classes=2)
        self.assertTrue(loss.ignore_index < 0)

        # ignore_last_class = True
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['ignore_last_class'] = True
        new_cfg_dict = solver_config_upgrader(old_cfg_dict, version=3)
        self.assertNotIn('ignore_last_class', new_cfg_dict)
        self.assertIn('ignore_class_index', new_cfg_dict)
        self.assertEqual(new_cfg_dict['ignore_class_index'], -1)

        new_cfg = build_config(new_cfg_dict)
        loss = new_cfg.build_loss(num_classes=2)
        self.assertEqual(loss.ignore_index, 1)

        # ignore_last_class = 'force'
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['ignore_last_class'] = 'force'
        new_cfg_dict = solver_config_upgrader(old_cfg_dict, version=3)
        self.assertNotIn('ignore_last_class', new_cfg_dict)
        self.assertIn('ignore_class_index', new_cfg_dict)
        self.assertEqual(new_cfg_dict['ignore_class_index'], -1)

        new_cfg = build_config(new_cfg_dict)
        loss = new_cfg.build_loss(num_classes=2)
        self.assertEqual(loss.ignore_index, 1)

    def test_disallow_loss_opts_if_external(self):
        args = dict(
            external_loss_def=ExternalModuleConfig(
                uri='abc/def', entrypoint='foo'),
            class_loss_weights=[1, 2])
        self.assertRaises(ValidationError, lambda: SolverConfig(**args))

        args = dict(
            external_loss_def=ExternalModuleConfig(
                uri='abc/def', entrypoint='foo'),
            ignore_class_index=1)
        self.assertRaises(ValidationError, lambda: SolverConfig(**args))

        args = dict(
            external_loss_def=ExternalModuleConfig(
                uri='abc/def', entrypoint='foo'))
        self.assertNoError(lambda: SolverConfig(**args))

    def test_build_loss(self):
        cfg = SolverConfig()
        loss = cfg.build_loss(num_classes=2)
        self.assertIsInstance(loss, nn.CrossEntropyLoss)

        cfg = SolverConfig(class_loss_weights=[1, 2])
        loss = cfg.build_loss(num_classes=2)
        self.assertListEqual(loss.weight.tolist(), [1, 2])

        cfg = SolverConfig(ignore_class_index=5)
        loss = cfg.build_loss(num_classes=10)
        self.assertEqual(loss.ignore_index, 5)

    def test_build(self):
        pass
