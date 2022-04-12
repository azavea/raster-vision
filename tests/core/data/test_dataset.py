from typing import Callable
import unittest

from rastervision.core.data import (DatasetConfig, ClassConfig, SceneConfig,
                                    RasterioSourceConfig, LabelSourceConfig,
                                    dataset_config_upgrader)
from rastervision.pipeline.config import ConfigError, build_config


def make_scene(id: str) -> SceneConfig:
    return SceneConfig(
        id=id,
        raster_source=RasterioSourceConfig(uris=['']),
        label_source=LabelSourceConfig())


class TestDatasetConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_all_scenes(self):
        class_config = ClassConfig(
            names=['red', 'green'], colors=['red', 'green'])

        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[],
            validation_scenes=[],
            test_scenes=[])
        self.assertEqual(len(cfg.all_scenes), 0)

        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene(str(i)) for i in range(1)],
            validation_scenes=[make_scene(str(i)) for i in range(2)],
            test_scenes=[make_scene(str(i)) for i in range(3)])
        self.assertEqual(len(cfg.all_scenes), 6)

    def test_default_scene_groups(self):
        """Test if DatasetConfig.update() correctly populates scene_groups
        with the default scene groups."""
        class_config = ClassConfig(
            names=['red', 'green'], colors=['red', 'green'])

        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene(str(i)) for i in range(1)],
            validation_scenes=[make_scene(str(i)) for i in range(2)],
            test_scenes=[make_scene(str(i)) for i in range(3)])
        cfg.update()

        all_train = {s.id for s in cfg.train_scenes}
        all_val = {s.id for s in cfg.validation_scenes}
        all_test = {s.id for s in cfg.test_scenes}
        self.assertEqual(cfg.scene_groups['train_scenes'], all_train)
        self.assertEqual(cfg.scene_groups['validation_scenes'], all_val)
        self.assertEqual(cfg.scene_groups['test_scenes'], all_test)

    def test_validate_config(self):
        """Test if DatasetConfig.update() correctly populates scene_groups
        with the default scene groups."""
        class_config = ClassConfig(
            names=['red', 'green'], colors=['red', 'green'])

        # non-unique train IDs
        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene('train'),
                          make_scene('train')],
            validation_scenes=[make_scene('val')])
        cfg.update()
        self.assertRaises(ConfigError, cfg.validate_config)

        # non-unique val + test IDs
        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene('train-1'),
                          make_scene('train-2')],
            validation_scenes=[make_scene('val')],
            test_scenes=[make_scene('val')])
        cfg.update()
        self.assertRaises(ConfigError, cfg.validate_config)

        # unknown IDs in scene group
        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[make_scene('train-1'),
                          make_scene('train-2')],
            validation_scenes=[make_scene('val')],
            test_scenes=[make_scene('test')],
            scene_groups={'abc': {'train-1', 'train-2', 'train-3'}})
        cfg.update()
        self.assertRaises(ConfigError, cfg.validate_config)

    def test_upgrader(self):
        class_config = ClassConfig(
            names=['red', 'green'], colors=['red', 'green'])

        cfg = DatasetConfig(
            class_config=class_config,
            train_scenes=[],
            validation_scenes=[],
            test_scenes=[])
        old_cfg_dict = cfg.dict()
        old_cfg_dict['img_channels'] = 8
        new_cfg_dict = dataset_config_upgrader(old_cfg_dict, version=0)
        self.assertNotIn('img_channels', new_cfg_dict)
        self.assertNoError(lambda: build_config(new_cfg_dict))


if __name__ == '__main__':
    unittest.main()
