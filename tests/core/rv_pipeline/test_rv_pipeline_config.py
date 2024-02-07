import unittest

from rastervision.core.data import (ClassConfig, DatasetConfig)
from rastervision.core.backend import (BackendConfig)
from rastervision.core.rv_pipeline.rv_pipeline_config import (
    PredictOptions, RVPipelineConfig, rv_pipeline_config_upgrader)


class TestPredictOptions(unittest.TestCase):
    def test_stride_validator(self):
        cfg = PredictOptions(chip_sz=10)
        self.assertEqual(cfg.stride, 10)


class TestRVPipelineConfig(unittest.TestCase):
    def test_upgrader(self):
        cfg_dict = dict(
            dataset=DatasetConfig(
                class_config=ClassConfig(names=[]),
                train_scenes=[],
                validation_scenes=[]),
            backend=BackendConfig(),
            train_chip_sz=20,
            chip_nodata_threshold=0.5,
            predict_chip_sz=20,
            predict_batch_sz=8)
        cfg_dict = rv_pipeline_config_upgrader(cfg_dict, 10)
        cfg_dict = rv_pipeline_config_upgrader(cfg_dict, 11)
        cfg = RVPipelineConfig(**cfg_dict)

        cfg_dict = dict(
            dataset=DatasetConfig(
                class_config=ClassConfig(names=[]),
                train_scenes=[],
                validation_scenes=[]),
            backend=BackendConfig(),
            train_chip_sz=20,
            chip_nodata_threshold=0.5,
            chip_options=dict(method='random'),
            predict_chip_sz=20,
            predict_batch_sz=8,
            predict_options=dict())
        cfg_dict = rv_pipeline_config_upgrader(cfg_dict, 10)
        cfg_dict = rv_pipeline_config_upgrader(cfg_dict, 11)
        cfg = RVPipelineConfig(**cfg_dict)
        self.assertEqual(cfg.chip_options.get_chip_sz(), 20)
        self.assertEqual(cfg.chip_options.nodata_threshold, 0.5)
        self.assertEqual(cfg.predict_options.chip_sz, 20)
        self.assertEqual(cfg.predict_options.batch_sz, 8)


if __name__ == '__main__':
    unittest.main()
