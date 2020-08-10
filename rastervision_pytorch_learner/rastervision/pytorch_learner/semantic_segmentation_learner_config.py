from enum import Enum
from typing import Union

from rastervision.pipeline.config import register_config, Field, validator
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)
from rastervision.pytorch_learner.learner_config import Backbone


class SemanticSegmentationDataFormat(Enum):
    default = 'default'


@register_config('semantic_segmentation_data')
class SemanticSegmentationDataConfig(DataConfig):
    data_format: SemanticSegmentationDataFormat = SemanticSegmentationDataFormat.default

    img_channels: int = Field(
        3, description='The number of channels of the training images.')

    img_format: str = Field(
        'png', description='The filetype of the training images.')
    label_format: str = Field(
        'png', description='The filetype of the training labels.')

    channel_display_groups: Union[dict, list, tuple] = Field(
        [(0, 1, 2)], description='Groups of image channels to display together.')


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
        from rastervision.pytorch_learner.semantic_segmentation_learner import (
            SemanticSegmentationLearner)
        return SemanticSegmentationLearner(
            self, tmp_dir, model_path=model_path)
