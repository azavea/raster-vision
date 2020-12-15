from typing import Callable
import unittest

from rastervision.pytorch_learner import (
    DataConfig, ImageDataConfig, SemanticSegmentationDataConfig,
    SemanticSegmentationImageDataConfig, ClassificationDataConfig,
    ClassificationImageDataConfig, RegressionDataConfig,
    RegressionImageDataConfig, ObjectDetectionDataConfig,
    ObjectDetectionImageDataConfig, data_config_upgrader,
    ss_data_config_upgrader, clf_data_config_upgrader,
    reg_data_config_upgrader, objdet_data_config_upgrader, GeoDataWindowConfig,
    GeoDataWindowMethod)
from rastervision.pipeline.config import (ConfigError, build_config)


class TestDataConfig(unittest.TestCase):
    def _test_config_upgrader(self, OldCfgType, NewCfgType, upgrader,
                              curr_version):
        old_cfg = OldCfgType()
        old_cfg_dict = old_cfg.dict()
        for i in range(curr_version):
            old_cfg_dict = upgrader(old_cfg_dict, version=i)
        new_cfg = build_config(old_cfg_dict)
        self.assertTrue(isinstance(new_cfg, NewCfgType))

    def test_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=DataConfig,
            NewCfgType=ImageDataConfig,
            upgrader=data_config_upgrader,
            curr_version=2)

    def test_ss_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=SemanticSegmentationDataConfig,
            NewCfgType=SemanticSegmentationImageDataConfig,
            upgrader=ss_data_config_upgrader,
            curr_version=2)

    def test_clf_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=ClassificationDataConfig,
            NewCfgType=ClassificationImageDataConfig,
            upgrader=clf_data_config_upgrader,
            curr_version=2)

    def test_reg_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=RegressionDataConfig,
            NewCfgType=RegressionImageDataConfig,
            upgrader=reg_data_config_upgrader,
            curr_version=2)

    def test_objdet_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=ObjectDetectionDataConfig,
            NewCfgType=ObjectDetectionImageDataConfig,
            upgrader=objdet_data_config_upgrader,
            curr_version=2)


class TestImageDataConfig(unittest.TestCase):
    def fail_if_fails(self, fn: Callable, msg=''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_group_uris(self):
        group_uris = ['a', 'b', 'c']

        # test missing group_uris
        cfg = ImageDataConfig(group_train_sz=1)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        cfg = ImageDataConfig(group_train_sz_rel=.5)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test both group_train_sz and group_train_sz_rel specified
        cfg = ImageDataConfig(
            group_uris=group_uris, group_train_sz=1, group_train_sz_rel=.5)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test length check
        cfg = ImageDataConfig(group_uris=group_uris, group_train_sz=[1])
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        cfg = ImageDataConfig(group_uris=group_uris, group_train_sz_rel=[.5])
        self.assertRaises(ConfigError, lambda: cfg.validate_config())
        # test valid configs
        cfg = ImageDataConfig(group_uris=group_uris, group_train_sz=1)
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = ImageDataConfig(
            group_uris=group_uris, group_train_sz=[1] * len(group_uris))
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = ImageDataConfig(group_uris=group_uris, group_train_sz_rel=.1)
        self.fail_if_fails(lambda: cfg.validate_config())
        cfg = ImageDataConfig(
            group_uris=group_uris, group_train_sz_rel=[.1] * len(group_uris))
        self.fail_if_fails(lambda: cfg.validate_config())


class TestGeoDataConfig(unittest.TestCase):
    def fail_if_fails(self, fn: Callable, msg=''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_window_config(self):
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding, size=10, stride=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(method=GeoDataWindowMethod.random, size=10)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=(10, 20))
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=None,
            w_lims=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=None,
            w_lims=(10, 20))
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())


if __name__ == '__main__':
    unittest.main()
