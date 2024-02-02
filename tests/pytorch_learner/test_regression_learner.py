from typing import Any, Callable
import unittest
from os.path import join
from uuid import uuid4

import numpy as np
import torch

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.data import (
    ClassConfig, DatasetConfig, RasterioSourceConfig, MultiRasterSourceConfig,
    ReclassTransformerConfig, SceneConfig, LabelSourceConfig)
from rastervision.core.rv_pipeline import (WindowSamplingConfig,
                                           WindowSamplingMethod)
from rastervision.pytorch_learner import (
    RegressionModelConfig, SolverConfig, RegressionGeoDataConfig,
    RegressionLearnerConfig, RegressionPlotOptions, RegressionLearner)
from tests import data_file_path


class MockRegressionabelSourceConfig(LabelSourceConfig):
    def build(self, *args, **kwargs):
        pass


def make_scene(num_channels: int, num_classes: int,
               tmp_dir: str) -> SceneConfig:
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
        raster_sources=rs_cfgs_img, channel_order=list(range(num_channels)))

    scene_cfg = SceneConfig(
        id=str(uuid4()),
        raster_source=rs_cfg_img,
        label_source=MockRegressionabelSourceConfig())
    return scene_cfg


class TestRegressionLearner(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_learner_rgb(self):
        args = dict(num_channels=3, channel_display_groups=None)
        self.assertNoError(lambda: self._test_learner(**args))

    def test_learner_multiband(self):
        args = dict(
            num_channels=6, channel_display_groups=[(0, 1, 2), (3, 4, 5)])
        self.assertNoError(lambda: self._test_learner(**args))

    def _test_learner(self,
                      num_channels: int,
                      channel_display_groups: Any,
                      num_classes: int = 5):
        """Tests learner init, plots, bundle, train and pred."""

        with get_tmp_dir() as tmp_dir:
            class_config = ClassConfig(
                names=[f'class_{i}' for i in range(num_classes)])
            dataset_cfg = DatasetConfig(
                class_config=class_config,
                train_scenes=[
                    make_scene(num_channels, num_classes, tmp_dir)
                    for _ in range(2)
                ],
                validation_scenes=[
                    make_scene(num_channels, num_classes, tmp_dir)
                    for _ in range(2)
                ],
                test_scenes=[])
            data_cfg = RegressionGeoDataConfig(
                scene_dataset=dataset_cfg,
                img_channels=num_channels,
                sampling=WindowSamplingConfig(
                    method=WindowSamplingMethod.random, size=20,
                    max_windows=8),
                class_names=class_config.names,
                class_colors=class_config.colors,
                plot_options=RegressionPlotOptions(
                    channel_display_groups=channel_display_groups),
                num_workers=0)

            learner_cfg = RegressionLearnerConfig(
                output_uri=tmp_dir,
                data=data_cfg,
                model=RegressionModelConfig(pretrained=False),
                solver=SolverConfig(batch_sz=4, num_epochs=1),
                log_tensorboard=False)

            learner = learner_cfg.build(tmp_dir, training=True)
            x = torch.rand((4, num_channels, 100, 100))
            y = torch.rand((4, num_classes))
            z = torch.rand((4, num_classes))
            learner.visualizer.plot_batch(x, y, join(tmp_dir, '1.png'))
            learner.visualizer.plot_batch(x, y, join(tmp_dir, '2.png'), z=z)

            learner.save_model_bundle()
            learner = RegressionLearner.from_model_bundle(
                learner.model_bundle_uri)


if __name__ == '__main__':
    unittest.main()
