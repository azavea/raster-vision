import unittest

from rastervision.pipeline.file_system import get_tmp_dir
from rastervision.pipeline.config import save_pipeline_config
from rastervision.core.data import (ClassConfig, DatasetConfig)
from rastervision.core.rv_pipeline import (
    SemanticSegmentationConfig, SemanticSegmentationChipOptions,
    SemanticSegmentationPredictOptions, WindowSamplingConfig,
    WindowSamplingMethod)
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (SemanticSegmentationModelConfig,
                                          SolverConfig)
from rastervision.pytorch_learner import (SemanticSegmentationGeoDataConfig,
                                          SemanticSegmentationImageDataConfig)

from tests.pytorch_learner.test_semantic_segmentation_learner import make_scene


def make_pipeline(tmp_dir: str, num_channels: int, nochip: bool = False):
    num_classes = 3
    class_config = ClassConfig(
        names=[f'class_{i}' for i in range(num_classes)])
    class_config.update()
    class_config.ensure_null_class()
    dataset_cfg = DatasetConfig(
        class_config=class_config,
        train_scenes=[
            make_scene(num_channels=num_channels, num_classes=num_classes)
            for _ in range(4)
        ],
        validation_scenes=[
            make_scene(num_channels=num_channels, num_classes=num_classes)
            for _ in range(2)
        ],
        test_scenes=[])
    chip_options = SemanticSegmentationChipOptions(
        sampling=WindowSamplingConfig(
            method=WindowSamplingMethod.random, size=100, max_windows=8))
    if nochip:
        data_cfg = SemanticSegmentationGeoDataConfig(
            scene_dataset=dataset_cfg,
            sampling=chip_options.sampling,
            num_workers=0)
    else:
        data_cfg = SemanticSegmentationImageDataConfig(num_workers=0)
    backend_cfg = PyTorchSemanticSegmentationConfig(
        data=data_cfg,
        model=SemanticSegmentationModelConfig(pretrained=False),
        solver=SolverConfig(batch_sz=4, num_epochs=1),
        log_tensorboard=False)
    pipeline_cfg = SemanticSegmentationConfig(
        root_uri=tmp_dir,
        dataset=dataset_cfg,
        backend=backend_cfg,
        chip_options=chip_options,
        predict_options=SemanticSegmentationPredictOptions(
            chip_sz=100, stride=50, crop_sz='auto'))
    pipeline_cfg.update()
    save_pipeline_config(pipeline_cfg, pipeline_cfg.get_config_uri())
    pipeline = pipeline_cfg.build(tmp_dir)
    return pipeline


class TestPyTorchSemanticSegmentation(unittest.TestCase):
    def test_full_chip_rgb(self):
        with get_tmp_dir() as tmp_dir:
            pipeline = make_pipeline(tmp_dir, 3)
            pipeline.chip()
            pipeline.train()
            pipeline.predict()
            pipeline.eval()
            pipeline.bundle()

    def test_full_chip_multispectral(self):
        with get_tmp_dir() as tmp_dir:
            pipeline = make_pipeline(tmp_dir, 4)
            pipeline.chip()
            pipeline.train()
            pipeline.predict()
            pipeline.eval()
            pipeline.bundle()

    def test_nochip_rgb(self):
        with get_tmp_dir() as tmp_dir:
            pipeline = make_pipeline(tmp_dir, 3, nochip=True)
            pipeline.train()

    def test_nochip_multispectral(self):
        with get_tmp_dir() as tmp_dir:
            pipeline = make_pipeline(tmp_dir, 4, nochip=True)
            pipeline.train()


if __name__ == '__main__':
    unittest.main()
