from typing import List

from rastervision.v2.core import PipelineConfig
from rastervision.v2.rv.data import DatasetConfig, ClassConfig
from rastervision.v2.rv.backend import BackendConfig

class TaskConfig(PipelineConfig):
    dataset: DatasetConfig
    backend: BackendConfig

    debug: bool = False
    train_chip_size: int = 200
    predict_chip_size: int = 800
    predict_batch_size: int = 8

    analyze_uri: str = None
    chip_uri: str = None
    train_uri: str = None
    predict_uri: str = None
    eval_uri: str = None

    def build(self):
        pass
