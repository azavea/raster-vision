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
    GeoDataWindowMethod, PlotOptions, ss_image_data_config_upgrader)
from rastervision.pipeline.config import (ConfigError, build_config)


class TestDataConfigToImageDataConfigUpgrade(unittest.TestCase):
    """Version 1 DataConfig should get upgraded to ImageDataConfigs."""

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
            curr_version=3)

    def test_ss_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=SemanticSegmentationDataConfig,
            NewCfgType=SemanticSegmentationImageDataConfig,
            upgrader=ss_data_config_upgrader,
            curr_version=3)

    def test_clf_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=ClassificationDataConfig,
            NewCfgType=ClassificationImageDataConfig,
            upgrader=clf_data_config_upgrader,
            curr_version=3)

    def test_reg_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=RegressionDataConfig,
            NewCfgType=RegressionImageDataConfig,
            upgrader=reg_data_config_upgrader,
            curr_version=3)

    def test_objdet_data_config_upgrader(self):
        self._test_config_upgrader(
            OldCfgType=ObjectDetectionDataConfig,
            NewCfgType=ObjectDetectionImageDataConfig,
            upgrader=objdet_data_config_upgrader,
            curr_version=3)


class TestDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_upgrader(self):
        old_cfg_dict = DataConfig().dict()
        del old_cfg_dict['img_channels']
        new_cfg_dict = data_config_upgrader(old_cfg_dict, version=2)
        self.assertNoError(lambda: build_config(new_cfg_dict))


class TestSemanticSegmentationDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_upgrader(self):
        old_cfg = SemanticSegmentationDataConfig()
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['img_channels'] = 8
        new_cfg_dict = ss_data_config_upgrader(old_cfg_dict, version=2)
        self.assertNotIn('img_channels', new_cfg_dict)
        self.assertNotIn('channel_display_groups', new_cfg_dict)
        self.assertNoError(lambda: build_config(new_cfg_dict))


class TestSemanticSegmentationImageDataConfig(unittest.TestCase):
    def test_upgrader(self):
        old_cfg = SemanticSegmentationImageDataConfig(img_channels=8)
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['channel_display_groups'] = None
        old_cfg_dict['img_format'] = 'npy'
        old_cfg_dict['label_format'] = 'npy'
        new_cfg_dict = ss_image_data_config_upgrader(old_cfg_dict, version=2)
        self.assertNotIn('channel_display_groups', new_cfg_dict)
        self.assertNotIn('img_format', new_cfg_dict)
        self.assertNotIn('label_format', new_cfg_dict)
        new_cfg: SemanticSegmentationImageDataConfig = build_config(
            new_cfg_dict)
        self.assertEqual(new_cfg.img_channels, old_cfg.img_channels)


class TestImageDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
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
        self.assertNoError(lambda: cfg.validate_config())
        cfg = ImageDataConfig(
            group_uris=group_uris, group_train_sz=[1] * len(group_uris))
        self.assertNoError(lambda: cfg.validate_config())
        cfg = ImageDataConfig(group_uris=group_uris, group_train_sz_rel=.1)
        self.assertNoError(lambda: cfg.validate_config())
        cfg = ImageDataConfig(
            group_uris=group_uris, group_train_sz_rel=[.1] * len(group_uris))
        self.assertNoError(lambda: cfg.validate_config())


class TestGeoDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_window_config(self):
        # require stride when method = sliding
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.sliding, size=10, stride=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        # update() corrrectly initializes size_lims
        cfg = GeoDataWindowConfig(method=GeoDataWindowMethod.random, size=10)
        cfg.update()
        self.assertEqual(cfg.size_lims, (10, 11))
        self.assertNoError(lambda: cfg.validate_config())

        # update() only initializes size_lims if method = random
        cfg = GeoDataWindowConfig(method=GeoDataWindowMethod.sliding, size=10)
        cfg.update()
        self.assertEqual(cfg.size_lims, None)

        # update() called by validate_config()
        cfg = GeoDataWindowConfig(method=GeoDataWindowMethod.random, size=10)
        self.assertNoError(lambda: cfg.validate_config())
        self.assertEqual(cfg.size_lims, (10, 11))

        # only allow on of size_lims and h_lims+w_lims
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=(10, 20))
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        # require both h_lims and w_lims if either specified
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=None,
            w_lims=(10, 20))
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        # require both h_lims and w_lims if either specified
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())

        # only allow on of size_lims and h_lims+w_lims
        cfg = GeoDataWindowConfig(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ConfigError, lambda: cfg.validate_config())


class TestPlotOptions(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_update(self):
        # without img_channels
        data_cfg = DataConfig()
        plot_opts = PlotOptions()
        self.assertNoError(lambda: plot_opts.update(data_cfg))

        # with img_channels
        data_cfg = DataConfig(img_channels=3)
        plot_opts = PlotOptions()
        self.assertNoError(lambda: plot_opts.update(data_cfg))

    def test_channel_display_groups(self):
        data_cfg = DataConfig(img_channels=3)
        # check error on empty dict
        opts = PlotOptions(channel_display_groups={})
        self.assertRaises(ConfigError, lambda: opts.update(data_cfg))

        # check error on empty list
        opts = PlotOptions(channel_display_groups=[])
        self.assertRaises(ConfigError, lambda: opts.update(data_cfg))

        # check error on group of size >3
        opts = PlotOptions(channel_display_groups={'a': list(range(4))})
        self.assertRaises(ConfigError, lambda: opts.update(data_cfg))

        # check error on out-of-range channel index
        opts = PlotOptions(channel_display_groups={'a': [0, 1, 10]})
        self.assertRaises(ConfigError, lambda: opts.update(data_cfg))

        # check auto conversion to dict
        data_cfg = DataConfig(img_channels=6)
        opts = PlotOptions(channel_display_groups=[(0, 1, 2), (4, 3, 5)])
        opts.update(data_cfg)
        self.assertIsInstance(opts.channel_display_groups, dict)
        self.assertIn('Channels: [0, 1, 2]', opts.channel_display_groups)
        self.assertIn('Channels: [4, 3, 5]', opts.channel_display_groups)
        self.assertListEqual(
            opts.channel_display_groups['Channels: [0, 1, 2]'], [0, 1, 2])
        self.assertListEqual(
            opts.channel_display_groups['Channels: [4, 3, 5]'], [4, 3, 5])


if __name__ == '__main__':
    unittest.main()
