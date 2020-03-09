from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)

data_formats = ['default']


@register_config('object_detection_data')
class ObjectDetectionDataConfig(DataConfig):
    data_format: str = 'default'

    def validate_data_format(self):
        self.validate_list('data_format', data_formats)


@register_config('object_detection_model')
class ObjectDetectionModelConfig(ModelConfig):
    pass


@register_config('object_detection_learner')
class ObjectDetectionLearnerConfig(LearnerConfig):
    data: ObjectDetectionDataConfig
    model: ObjectDetectionModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.object_detection_learner import (
            ObjectDetectionLearner)
        return ObjectDetectionLearner(self, tmp_dir, model_path=model_path)
