from enum import Enum
from typing import Union, Optional
from pydantic import PositiveInt

from rastervision.pipeline.config import register_config, Field, validator
from rastervision.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)
from rastervision.pytorch_learner.learner_config import Backbone


class SemanticSegmentationDataFormat(Enum):
    default = 'default'


@register_config('semantic_segmentation_data')
class SemanticSegmentationDataConfig(DataConfig):
    data_format: SemanticSegmentationDataFormat = SemanticSegmentationDataFormat.default

    img_channels: PositiveInt = Field(
        3, description='The number of channels of the training images.')

    img_format: Optional[str] = Field(
        None, description='The filetype of the training images.')
    label_format: str = Field(
        'png', description='The filetype of the training labels.')

    channel_display_groups: Optional[Union[dict, list, tuple]] = Field(
        None, description='Groups of image channels to display together.')

    def update(self, **kwargs):
        super().update()

        if self.img_format is None:
            self.img_format = 'png' if self.img_channels == 3 else 'npy'

        if self.channel_display_groups is None:
            self.channel_display_groups = {
                'Input': tuple(range(self.img_channels))
            }


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
