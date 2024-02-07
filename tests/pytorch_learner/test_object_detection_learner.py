from typing import Any, Callable
from os.path import join
import unittest
from uuid import uuid4

import numpy as np

from rastervision.pipeline.file_system import get_tmp_dir, json_to_file
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassConfig, ClassInferenceTransformerConfig, DatasetConfig,
    GeoJSONVectorSourceConfig, geoms_to_geojson, pixel_to_map_coords,
    RasterioCRSTransformer, RasterioSourceConfig, MultiRasterSourceConfig,
    ReclassTransformerConfig, SceneConfig, ObjectDetectionLabelSourceConfig)
from rastervision.core.rv_pipeline import (ObjectDetectionConfig,
                                           ObjectDetectionWindowSamplingConfig,
                                           WindowSamplingMethod)
from rastervision.pytorch_backend import PyTorchObjectDetectionConfig
from rastervision.pytorch_learner import (
    ObjectDetectionModelConfig, SolverConfig, ObjectDetectionGeoDataConfig,
    PlotOptions, Backbone)
from tests import data_file_path


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

    extent = Box(0, 0, 600, 600)
    geoms = [extent.make_random_square(20).to_shapely() for _ in range(20)]
    props = [dict(class_id=np.random.randint(0, num_classes)) for _ in geoms]
    geojson = geoms_to_geojson(geoms, properties=props)
    geojson = pixel_to_map_coords(geojson,
                                  RasterioCRSTransformer.from_uri(path))
    label_uri = join(tmp_dir, 'labels.json')
    json_to_file(geojson, label_uri)

    label_source_cfg = ObjectDetectionLabelSourceConfig(
        vector_source=GeoJSONVectorSourceConfig(
            uris=label_uri,
            transformers=[ClassInferenceTransformerConfig(
                default_class_id=0)]))
    scene_cfg = SceneConfig(
        id=str(uuid4()),
        raster_source=rs_cfg_img,
        label_source=label_source_cfg)
    return scene_cfg


class TestObjectDetectionLearner(unittest.TestCase):
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
            data_cfg = ObjectDetectionGeoDataConfig(
                scene_dataset=dataset_cfg,
                sampling=ObjectDetectionWindowSamplingConfig(
                    method=WindowSamplingMethod.random,
                    size=200,
                    max_windows=8,
                    neg_ratio=0.5),
                class_names=class_config.names,
                class_colors=class_config.colors,
                plot_options=PlotOptions(
                    channel_display_groups=channel_display_groups),
                num_workers=0)
            backend_cfg = PyTorchObjectDetectionConfig(
                data=data_cfg,
                model=ObjectDetectionModelConfig(
                    backbone=Backbone.resnet18, pretrained=False),
                solver=SolverConfig(batch_sz=4, num_epochs=1),
                log_tensorboard=False)
            pipeline_cfg = ObjectDetectionConfig(
                root_uri=tmp_dir, dataset=dataset_cfg, backend=backend_cfg)
            pipeline_cfg.update()
            backend = backend_cfg.build(pipeline_cfg, tmp_dir)
            learner = backend.learner_cfg.build(tmp_dir, training=True)

            learner.plot_dataloaders()
            learner.train()
            learner.plot_predictions(split='valid')
            learner.save_model_bundle()


if __name__ == '__main__':
    unittest.main()
