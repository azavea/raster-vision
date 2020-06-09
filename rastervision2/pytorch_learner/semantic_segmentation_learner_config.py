from enum import Enum

from rastervision2.pipeline.config import Field, register_config, validator
from rastervision2.pytorch_learner.learner_config import (Backbone, DataConfig,
                                                          LearnerConfig,
                                                          ModelConfig)


class SemanticSegmentationDataFormat(Enum):
    default = 1


@register_config('semantic_segmentation_data')
class SemanticSegmentationDataConfig(DataConfig):
    data_format: SemanticSegmentationDataFormat = SemanticSegmentationDataFormat.default


@register_config('semantic_segmentation_model')
class SemanticSegmentationModelConfig(ModelConfig):
    backbone: Backbone = Field(
        Backbone.resnet50,
        description=(
            'The torchvision.models backbone to use. At the moment only '
            'resnet50 or resnet101 will work.'))

    @validator('backbone')
    def only_valid_backbones(cls, v):
        if v not in [Backbone.resnet50, Backbone.resnet101]:
            raise ValueError(
                'The only valid backbones for DeepLabv3 are resnet50 '
                'and resnet101.')
        return v


@register_config('semantic_segmentation_learner')
class SemanticSegmentationLearnerConfig(LearnerConfig):
    data: SemanticSegmentationDataConfig
    model: SemanticSegmentationModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.semantic_segmentation_learner import (
            SemanticSegmentationLearner)
        return SemanticSegmentationLearner(
            self, tmp_dir, model_path=model_path)
