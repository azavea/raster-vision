from enum import Enum

from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)


class DataFormat(Enum):
    default = 1


@register_config('semantic_segmentation_data')
class SemanticSegmentationDataConfig(DataConfig):
    data_format: DataFormat = DataFormat.default


@register_config('semantic_segmentation_model')
class SemanticSegmentationModelConfig(ModelConfig):
    pass


@register_config('semantic_segmentation_learner')
class SemanticSegmentationLearnerConfig(LearnerConfig):
    data: SemanticSegmentationDataConfig
    model: SemanticSegmentationModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.semantic_segmentation_learner import (
            SemanticSegmentationLearner)
        return SemanticSegmentationLearner(
            self, tmp_dir, model_path=model_path)
