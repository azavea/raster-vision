from typing import Callable
import unittest

from rastervision.pytorch_learner.learner_config import DataConfig
from rastervision.pipeline.config import ConfigError


class TestDataConfig(unittest.TestCase):
    def fail_if_fails(self, fn: Callable, msg=''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_group_uris(self):
        group_uris = ['a', 'b', 'c']

        # test missing group_uris
        cfg = DataConfig(group_train_sz=1)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        cfg = DataConfig(group_train_sz_rel=.5)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test both group_train_sz and group_train_sz_rel specified
        cfg = DataConfig(
            group_uris=group_uris, group_train_sz=1, group_train_sz_rel=.5)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test length check
        cfg = DataConfig(group_uris=group_uris, group_train_sz=[1])
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        cfg = DataConfig(group_uris=group_uris, group_train_sz_rel=[.5])
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test valid configs
        cfg = DataConfig(group_uris=group_uris, group_train_sz=1)
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = DataConfig(
            group_uris=group_uris, group_train_sz=[1] * len(group_uris))
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = DataConfig(group_uris=group_uris, group_train_sz_rel=.1)
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = DataConfig(
            group_uris=group_uris, group_train_sz_rel=[.1] * len(group_uris))
        self.fail_if_fails(lambda: cfg.validate_config())


if __name__ == '__main__':
    unittest.main()
