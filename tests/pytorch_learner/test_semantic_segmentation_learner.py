from typing import Any, Callable
import unittest
from uuid import uuid4

import numpy as np

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.core.data import (
    ClassConfig, DatasetConfig, RasterioSourceConfig, MultiRasterSourceConfig,
    ReclassTransformerConfig, SceneConfig,
    SemanticSegmentationLabelSourceConfig)
from rastervision.core.rv_pipeline import (
    SemanticSegmentationConfig, WindowSamplingConfig, WindowSamplingMethod)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (
    SemanticSegmentationModelConfig, SolverConfig,
    SemanticSegmentationGeoDataConfig, PlotOptions)
from rastervision.pytorch_learner.utils import (
    serialize_albumentation_transform)
from tests.data_files.lambda_transforms import lambda_transforms
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
        raster_sources=rs_cfgs_img, channel_order=list(range(num_channels)))
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


class TestSemanticSegmentationLearner(unittest.TestCase):
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
            class_config.update()
            class_config.ensure_null_class()
            dataset_cfg = DatasetConfig(
                class_config=class_config,
                train_scenes=[
                    make_scene(
                        num_channels=num_channels, num_classes=num_classes)
                    for _ in range(4)
                ],
                validation_scenes=[
                    make_scene(
                        num_channels=num_channels, num_classes=num_classes)
                    for _ in range(2)
                ],
                test_scenes=[])
            if num_channels == 6:
                tf = lambda_transforms['swap']
                aug_tf = serialize_albumentation_transform(
                    tf,
                    lambda_transforms_path=data_file_path(
                        'lambda_transforms.py'),
                    dst_dir=tmp_dir)
            else:
                aug_tf = None
            data_cfg = SemanticSegmentationGeoDataConfig(
                scene_dataset=dataset_cfg,
                sampling=WindowSamplingConfig(
                    method=WindowSamplingMethod.random, size=20,
                    max_windows=8),
                class_config=class_config,
                aug_transform=aug_tf,
                plot_options=PlotOptions(
                    channel_display_groups=channel_display_groups),
                num_workers=0)
            backend_cfg = PyTorchSemanticSegmentationConfig(
                data=data_cfg,
                model=SemanticSegmentationModelConfig(pretrained=False),
                solver=SolverConfig(
                    batch_sz=4, num_epochs=1, ignore_class_index=-1),
                log_tensorboard=False)
            pipeline_cfg = SemanticSegmentationConfig(
                root_uri=tmp_dir, dataset=dataset_cfg, backend=backend_cfg)
            pipeline_cfg.update()

            backend = backend_cfg.build(pipeline_cfg, tmp_dir)
            learner = backend.learner_cfg.build(tmp_dir, training=True)

            learner.plot_dataloaders()
            learner.train(1)
            learner.train(1)
            learner.plot_predictions(split='valid')
            learner.save_model_bundle()


if __name__ == '__main__':
    unittest.main()
