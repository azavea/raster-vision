from typing import Any, Callable
import unittest
from os.path import join
from uuid import uuid4

import numpy as np

from rastervision.pipeline.file_system import json_to_file, get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data import (
    ClassConfig, DatasetConfig, geoms_to_geojson, pixel_to_map_coords,
    RasterioCRSTransformer, RasterioSourceConfig, MultiRasterSourceConfig,
    ReclassTransformerConfig, SceneConfig, ChipClassificationLabelSourceConfig,
    GeoJSONVectorSourceConfig)
from rastervision.core.rv_pipeline import (ChipClassificationConfig,
                                           ChipOptions, WindowSamplingConfig,
                                           WindowSamplingMethod)
from rastervision.pytorch_backend import PyTorchChipClassificationConfig
from rastervision.pytorch_learner import (
    ClassificationModelConfig, SolverConfig, ClassificationGeoDataConfig,
    PlotOptions)
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

    geoms = [b.to_shapely() for b in Box(0, 0, 600, 600).get_windows(100, 100)]
    props = [dict(class_id=np.random.randint(0, num_classes)) for _ in geoms]
    geojson = geoms_to_geojson(geoms, properties=props)
    geojson = pixel_to_map_coords(geojson,
                                  RasterioCRSTransformer.from_uri(path))
    uri = join(tmp_dir, 'labels.json')
    json_to_file(geojson, uri)

    label_source_cfg = ChipClassificationLabelSourceConfig(
        vector_source=GeoJSONVectorSourceConfig(uris=uri),
        background_class_id=0,
        use_intersection_over_cell=True)
    scene_cfg = SceneConfig(
        id=str(uuid4()),
        raster_source=rs_cfg_img,
        label_source=label_source_cfg)
    return scene_cfg


class TestClassificationLearner(unittest.TestCase):
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
            sampling_cfg = WindowSamplingConfig(
                method=WindowSamplingMethod.random, size=20, max_windows=8)
            data_cfg = ClassificationGeoDataConfig(
                scene_dataset=dataset_cfg,
                sampling=sampling_cfg,
                class_names=class_config.names,
                class_colors=class_config.colors,
                plot_options=PlotOptions(
                    channel_display_groups=channel_display_groups),
                num_workers=0)
            backend_cfg = PyTorchChipClassificationConfig(
                data=data_cfg,
                model=ClassificationModelConfig(pretrained=False),
                solver=SolverConfig(batch_sz=4, num_epochs=1),
                log_tensorboard=False)
            pipeline_cfg = ChipClassificationConfig(
                root_uri=tmp_dir,
                dataset=dataset_cfg,
                backend=backend_cfg,
                chip_options=ChipOptions(sampling=sampling_cfg))
            pipeline_cfg.update()
            backend = backend_cfg.build(pipeline_cfg, tmp_dir)
            learner = backend.learner_cfg.build(tmp_dir, training=True)

            learner.plot_dataloaders()
            learner.train()
            learner.plot_predictions(split='valid')
            learner.save_model_bundle()

            learner = None
            backend.learner = None
            backend.load_model()

            pred_scene = dataset_cfg.validation_scenes[0].build(
                class_config, tmp_dir)
            _ = backend.predict_scene(pred_scene, chip_sz=100)


if __name__ == '__main__':
    unittest.main()
