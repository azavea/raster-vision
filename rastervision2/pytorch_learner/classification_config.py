from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_learner.learner_config import (
    LearnerConfig, DataConfig, ModelConfig)

data_formats = ['image_folder']


@register_config('classification_data')
class ClassificationDataConfig(DataConfig):
    data_format: str = 'image_folder'

    def validate_data_format(self):
        self.validate_list('data_format', data_formats)


@register_config('classification_model')
class ClassificationModelConfig(ModelConfig):
    pass

@register_config('classification_learner')
class ClassificationLearnerConfig(LearnerConfig):
    data: ClassificationDataConfig
    model: ClassificationModelConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.classification_learner import (
            ClassificationLearner)
        return ClassificationLearner(self, tmp_dir, model_path=model_path)
