from typing import Callable
import unittest

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.pipeline.config import (ValidationError, build_config)
from rastervision.pytorch_learner import (
    DataConfig, ImageDataConfig, SemanticSegmentationDataConfig,
    SemanticSegmentationImageDataConfig, SemanticSegmentationGeoDataConfig,
    ClassificationDataConfig, ClassificationImageDataConfig,
    RegressionDataConfig, RegressionImageDataConfig, ObjectDetectionDataConfig,
    ObjectDetectionImageDataConfig, data_config_upgrader,
    ss_data_config_upgrader, clf_data_config_upgrader,
    reg_data_config_upgrader, objdet_data_config_upgrader, GeoDataWindowConfig,
    GeoDataWindowMethod, GeoDataConfig, PlotOptions,
    ss_image_data_config_upgrader)
from rastervision.core.data import DatasetConfig, ClassConfig


class TestDataConfigToImageDataConfigUpgrade(unittest.TestCase):
    """Version 1 DataConfig should get upgraded to ImageDataConfigs."""

    def _test_config_upgrader(self, old_cfg_type: type, new_cfg_type: type,
                              upgrader: Callable, curr_version: int):
        old_cfg = old_cfg_type()
        old_cfg_dict = old_cfg.dict()
        for i in range(curr_version):
            old_cfg_dict = upgrader(old_cfg_dict, version=i)
        new_cfg = build_config(old_cfg_dict)
        self.assertTrue(isinstance(new_cfg, new_cfg_type))

    def test_data_config_upgrader(self):
        self._test_config_upgrader(
            old_cfg_type=DataConfig,
            new_cfg_type=ImageDataConfig,
            upgrader=data_config_upgrader,
            curr_version=3)

    def test_ss_data_config_upgrader(self):
        self._test_config_upgrader(
            old_cfg_type=SemanticSegmentationDataConfig,
            new_cfg_type=SemanticSegmentationImageDataConfig,
            upgrader=ss_data_config_upgrader,
            curr_version=3)

    def test_clf_data_config_upgrader(self):
        self._test_config_upgrader(
            old_cfg_type=ClassificationDataConfig,
            new_cfg_type=ClassificationImageDataConfig,
            upgrader=clf_data_config_upgrader,
            curr_version=3)

    def test_reg_data_config_upgrader(self):
        self._test_config_upgrader(
            old_cfg_type=RegressionDataConfig,
            new_cfg_type=RegressionImageDataConfig,
            upgrader=reg_data_config_upgrader,
            curr_version=3)

    def test_objdet_data_config_upgrader(self):
        self._test_config_upgrader(
            old_cfg_type=ObjectDetectionDataConfig,
            new_cfg_type=ObjectDetectionImageDataConfig,
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

    def test_build_cc(self):
        import os
        from os.path import join
        import numpy as np
        from rastervision.pytorch_backend.pytorch_learner_backend import (
            get_image_ext, write_chip)
        from rastervision.pipeline.file_system import zipdir

        nclasses = 2
        class_names = [f'class_{i}' for i in range(nclasses)]
        chip_sz = 100
        img_sz = 200
        nchannels = 3
        nchips = 5
        with get_tmp_dir() as tmp_dir:
            # prepare data
            data_dir = join(tmp_dir, 'data')
            for split in ['train', 'valid']:
                os.makedirs(join(data_dir, split))
                for c in class_names:
                    class_dir = join(data_dir, split, c)
                    os.makedirs(class_dir)
                    for i in range(nchips):
                        chip = np.random.randint(
                            0,
                            256,
                            size=(chip_sz, chip_sz, nchannels),
                            dtype=np.uint8)
                        ext = get_image_ext(chip)
                        path = join(class_dir, f'{i}.{ext}')
                        write_chip(chip, path)

            # data config -- unzipped
            data_cfg = ClassificationImageDataConfig(
                uri=data_dir,
                class_names=class_names,
                img_channels=nchannels,
                img_sz=img_sz)
            train_ds, val_ds, test_ds = data_cfg.build(tmp_dir)
            self.assertEqual(len(train_ds), nclasses * nchips)
            self.assertEqual(len(val_ds), nclasses * nchips)
            self.assertEqual(len(test_ds), 0)
            x, y = train_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertIn(y, range(nclasses))
            x, y = val_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertIn(y, range(nclasses))
            del train_ds
            del val_ds
            del test_ds

            # data config -- zipped
            zip_path = join(tmp_dir, 'data.zip')
            zipdir(data_dir, zip_path)
            data_cfg = ClassificationImageDataConfig(
                uri=zip_path,
                class_names=class_names,
                img_channels=nchannels,
                img_sz=img_sz)
            train_ds, val_ds, test_ds = data_cfg.build(tmp_dir)
            self.assertEqual(len(train_ds), nclasses * nchips)
            self.assertEqual(len(val_ds), nclasses * nchips)
            self.assertEqual(len(test_ds), 0)
            x, y = train_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertIn(y, range(nclasses))
            x, y = val_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertIn(y, range(nclasses))
            del train_ds
            del val_ds
            del test_ds

    def test_build_ss(self):
        import os
        from os.path import join
        import numpy as np
        from rastervision.pytorch_backend.pytorch_learner_backend import (
            write_chip)

        nclasses = 2
        class_names = [f'class_{i}' for i in range(nclasses)]
        chip_sz = 100
        img_sz = 200
        nchannels = 3
        nchips = 5
        with get_tmp_dir() as tmp_dir:
            # prepare data
            data_dir = join(tmp_dir, 'data')
            for split in ['train', 'valid']:
                os.makedirs(join(data_dir, split))
                img_dir = join(data_dir, split, 'img')
                label_dir = join(data_dir, split, 'labels')
                os.makedirs(img_dir)
                os.makedirs(label_dir)
                for i in range(nchips):
                    chip = np.random.randint(
                        0,
                        256,
                        size=(chip_sz, chip_sz, nchannels),
                        dtype=np.uint8)
                    label = (chip[..., 0] > 128).astype(np.uint8)
                    img_path = join(img_dir, f'{i}.npy')
                    label_path = join(label_dir, f'{i}.npy')
                    write_chip(chip, img_path)
                    write_chip(label, label_path)

            # data config -- unzipped
            data_cfg = SemanticSegmentationImageDataConfig(
                uri=data_dir,
                class_names=class_names,
                img_channels=nchannels,
                img_sz=img_sz)
            train_ds, val_ds, test_ds = data_cfg.build(tmp_dir)
            self.assertEqual(len(train_ds), nchips)
            self.assertEqual(len(val_ds), nchips)
            self.assertEqual(len(test_ds), 0)
            x, y = train_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertEqual(y.shape, (img_sz, img_sz))
            x, y = val_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertEqual(y.shape, (img_sz, img_sz))
            del train_ds
            del val_ds
            del test_ds


class TestGeoDataConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_window_config(self):
        # update() correctly initializes size_lims
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

    def test_get_class_info_from_class_config_if_needed(self):
        class_config = ClassConfig(names=['bg', 'fg'])
        scene_dataset = DatasetConfig(
            class_config=class_config, train_scenes=[], validation_scenes=[])
        args = dict(scene_dataset=scene_dataset, window_opts={})
        self.assertNoError(lambda: GeoDataConfig(**args))

        cfg = GeoDataConfig(**args)
        self.assertListEqual(cfg.class_names, class_config.names)
        self.assertListEqual(cfg.class_colors, class_config.colors)

    def test_build_ss(self):
        from uuid import uuid4
        import numpy as np
        from rastervision.core.data import (
            ClassConfig, DatasetConfig, RasterioSourceConfig,
            MultiRasterSourceConfig, ReclassTransformerConfig, SceneConfig,
            SemanticSegmentationLabelSourceConfig)
        from tests import data_file_path

        def make_scene(num_channels: int, num_classes: int) -> SceneConfig:
            path = data_file_path('multi_raster_source/const_100_600x600.tiff')
            rs_cfgs_img = []
            for _ in range(num_channels):
                rs_cfg = RasterioSourceConfig(
                    uris=[path],
                    channel_order=[0],
                    transformers=[
                        ReclassTransformerConfig(
                            mapping={100: np.random.randint(0, 256)})
                    ])
                rs_cfgs_img.append(rs_cfg)
            rs_cfg_img = MultiRasterSourceConfig(
                raster_sources=rs_cfgs_img,
                channel_order=list(range(num_channels)))
            rs_cfg_label = RasterioSourceConfig(
                uris=[path],
                channel_order=[0],
                transformers=[
                    ReclassTransformerConfig(
                        mapping={100: np.random.randint(0, num_classes)})
                ])
            scene_cfg = SceneConfig(
                id=str(uuid4()),
                raster_source=rs_cfg_img,
                label_source=SemanticSegmentationLabelSourceConfig(
                    raster_source=rs_cfg_label))
            return scene_cfg

        nclasses = 2
        nchannels = 3
        chip_sz = 100
        img_sz = 200
        class_config = ClassConfig(
            names=[f'class_{i}' for i in range(nclasses)],
            null_class='class_0')
        dataset_cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene(nchannels, nclasses) for _ in range(4)],
            validation_scenes=[
                make_scene(nchannels, nclasses) for _ in range(2)
            ],
            test_scenes=[make_scene(nchannels, nclasses) for _ in range(0)])
        data_cfg = SemanticSegmentationGeoDataConfig(
            scene_dataset=dataset_cfg,
            window_opts=GeoDataWindowConfig(
                size=chip_sz, stride=chip_sz, padding=0),
            class_names=class_config.names,
            class_colors=class_config.colors,
            img_sz=img_sz,
            num_workers=0)
        with get_tmp_dir() as tmp_dir:
            train_ds, val_ds, test_ds = data_cfg.build(tmp_dir)
            self.assertEqual(len(train_ds), 4 * (600 // chip_sz)**2)
            self.assertEqual(len(val_ds), 2 * (600 // chip_sz)**2)
            self.assertEqual(len(test_ds), 0 * (600 // chip_sz)**2)
            x, y = train_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertEqual(y.shape, (img_sz, img_sz))
            x, y = val_ds[0]
            self.assertEqual(x.shape, (nchannels, img_sz, img_sz))
            self.assertEqual(y.shape, (img_sz, img_sz))
            del train_ds
            del val_ds
            del test_ds


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
