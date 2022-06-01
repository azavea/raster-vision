from typing import Callable
import unittest

from rastervision.pytorch_learner import (
    DataConfig, ImageDataConfig, SemanticSegmentationDataConfig,
    SemanticSegmentationImageDataConfig, SemanticSegmentationGeoDataConfig,
    ClassificationDataConfig, ClassificationImageDataConfig,
    RegressionDataConfig, RegressionImageDataConfig, ObjectDetectionDataConfig,
    ObjectDetectionImageDataConfig, data_config_upgrader,
    ss_data_config_upgrader, clf_data_config_upgrader,
    reg_data_config_upgrader, objdet_data_config_upgrader, GeoDataWindowConfig,
    GeoDataWindowMethod, PlotOptions, ss_image_data_config_upgrader)
from rastervision.pipeline.config import (ValidationError, build_config)
from rastervision.core.data import DatasetConfig, ClassConfig


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
        old_cfg_dict['channel_display_groups'] = None
        new_cfg_dict = ss_data_config_upgrader(old_cfg_dict, version=2)
        self.assertNotIn('channel_display_groups', new_cfg_dict)
        self.assertNoError(lambda: build_config(new_cfg_dict))


class TestSemanticSegmentationGeoDataConfig(unittest.TestCase):
    def test_upgrader(self):
        scene_dataset = DatasetConfig(
            train_scenes=[],
            validation_scenes=[],
            class_config=ClassConfig(names=['abc']))
        old_cfg = SemanticSegmentationGeoDataConfig(
            scene_dataset=scene_dataset,
            window_opts=GeoDataWindowConfig(size=100),
            img_channels=8)
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['channel_display_groups'] = None

        old_cfg_dict = data_config_upgrader(old_cfg_dict, version=2)
        new_cfg_dict = ss_data_config_upgrader(old_cfg_dict, version=2)

        self.assertNotIn('channel_display_groups', new_cfg_dict)
        new_cfg: SemanticSegmentationGeoDataConfig = build_config(new_cfg_dict)
        self.assertEqual(new_cfg.img_channels, 8)


class TestSemanticSegmentationImageDataConfig(unittest.TestCase):
    def test_upgrader(self):
        old_cfg = SemanticSegmentationImageDataConfig(img_channels=8)
        old_cfg_dict = old_cfg.dict()
        old_cfg_dict['channel_display_groups'] = None
        old_cfg_dict['img_format'] = 'npy'
        old_cfg_dict['label_format'] = 'npy'

        old_cfg_dict = data_config_upgrader(old_cfg_dict, version=2)
        old_cfg_dict = ss_data_config_upgrader(old_cfg_dict, version=2)
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
        args = dict(group_train_sz=1)
        self.assertRaises(ValidationError, lambda: ImageDataConfig(**args))
        args = dict(group_train_sz_rel=.5)
        self.assertRaises(ValidationError, lambda: ImageDataConfig(**args))
        # test both group_train_sz and group_train_sz_rel specified
        args = dict(
            group_uris=group_uris, group_train_sz=1, group_train_sz_rel=.5)
        self.assertRaises(ValidationError, lambda: ImageDataConfig(**args))
        # test length check
        args = dict(group_uris=group_uris, group_train_sz=[1])
        self.assertRaises(ValidationError, lambda: ImageDataConfig(**args))
        args = dict(group_uris=group_uris, group_train_sz_rel=[.5])
        self.assertRaises(ValidationError, lambda: ImageDataConfig(**args))
        # test valid configs
        args = dict(group_uris=group_uris, group_train_sz=1)
        self.assertNoError(lambda: ImageDataConfig(**args))
        args = dict(
            group_uris=group_uris, group_train_sz=[1] * len(group_uris))
        self.assertNoError(lambda: ImageDataConfig(**args))
        args = dict(group_uris=group_uris, group_train_sz_rel=.1)
        self.assertNoError(lambda: ImageDataConfig(**args))
        args = dict(
            group_uris=group_uris, group_train_sz_rel=[.1] * len(group_uris))
        self.assertNoError(lambda: ImageDataConfig(**args))


class TestGeoDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_window_config(self):
        # update() corrrectly initializes size_lims
        args = dict(method=GeoDataWindowMethod.random, size=10)
        self.assertNoError(lambda: GeoDataWindowConfig(**args))
        self.assertEqual(GeoDataWindowConfig(**args).size_lims, (10, 11))

        # update() only initializes size_lims if method = random
        args = dict(method=GeoDataWindowMethod.sliding, size=10)
        self.assertEqual(GeoDataWindowConfig(**args).size_lims, None)

        # only allow one of size_lims and h_lims+w_lims
        args = dict(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=(10, 20))
        self.assertRaises(ValidationError, lambda: GeoDataWindowConfig(**args))

        # require both h_lims and w_lims if either specified
        args = dict(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=None,
            w_lims=(10, 20))
        self.assertRaises(ValidationError, lambda: GeoDataWindowConfig(**args))

        # require both h_lims and w_lims if either specified
        args = dict(
            method=GeoDataWindowMethod.random,
            size=10,
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ValidationError, lambda: GeoDataWindowConfig(**args))

        # only allow one of size_lims and h_lims+w_lims
        args = dict(
            method=GeoDataWindowMethod.random,
            size=10,
            size_lims=(10, 20),
            h_lims=(10, 20),
            w_lims=None)
        self.assertRaises(ValidationError, lambda: GeoDataWindowConfig(**args))


class TestPlotOptions(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_update(self):
        # without img_channels
        plot_opts = PlotOptions()
        self.assertNoError(lambda: plot_opts.update(img_channels=None))

        # with img_channels
        plot_opts = PlotOptions()
        self.assertNoError(lambda: plot_opts.update(img_channels=3))

    def test_channel_display_groups(self):
        # check error on empty dict
        args = dict(channel_display_groups={})
        self.assertRaises(ValidationError, lambda: PlotOptions(**args))

        # check error on empty list
        args = dict(channel_display_groups=[])
        self.assertRaises(ValidationError, lambda: PlotOptions(**args))

        # check error on group of size >3
        # check error on empty list
        args = dict(channel_display_groups={'a': list(range(4))})
        self.assertRaises(ValidationError, lambda: PlotOptions(**args))

        # check auto conversion to dict
        data_cfg = DataConfig(
            img_channels=6,
            plot_options=PlotOptions(channel_display_groups=[(0, 1, 2), (4, 3,
                                                                         5)]))
        opts = data_cfg.plot_options
        self.assertIsInstance(opts.channel_display_groups, dict)
        self.assertIn('Channels: [0, 1, 2]', opts.channel_display_groups)
        self.assertIn('Channels: [4, 3, 5]', opts.channel_display_groups)
        self.assertListEqual(
            opts.channel_display_groups['Channels: [0, 1, 2]'], [0, 1, 2])
        self.assertListEqual(
            opts.channel_display_groups['Channels: [4, 3, 5]'], [4, 3, 5])


if __name__ == '__main__':
    unittest.main()
