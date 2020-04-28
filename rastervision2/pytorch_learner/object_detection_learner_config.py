from enum import Enum

from rastervision2.pipeline.config import register_config, Field, validator
from rastervision2.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig, Backbone)


class ObjectDetectionDataFormat(Enum):
    default = 1


@register_config('object_detection_data')
class ObjectDetectionDataConfig(DataConfig):
    data_format: ObjectDetectionDataFormat = ObjectDetectionDataFormat.default


@register_config('object_detection_model')
class ObjectDetectionModelConfig(ModelConfig):
    backbone: Backbone = Field(
        Backbone.resnet50,
        description=(
            'The torchvision.models backbone to use. At the moment only '
            'resnet50 will work.'))

    @validator('backbone')
    def only_valid_backbones(cls, v):
        if v not in [Backbone.resnet50]:
            raise ValueError(
                'The only valid backbone for Faster-RCNN is resnet50.')
        return v


@register_config('object_detection_learner')
class ObjectDetectionLearnerConfig(LearnerConfig):
    data: ObjectDetectionDataConfig
    model: ObjectDetectionModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.object_detection_learner import (
            ObjectDetectionLearner)
        return ObjectDetectionLearner(self, tmp_dir, model_path=model_path)
