from typing import List

from rastervision.pipeline.config import register_config, Field
from rastervision.core.backend import BackendConfig
from rastervision.pytorch_learner.learner_config import (
    SolverConfig, ModelConfig, default_augmentors, augmentors as
    augmentor_list)


@register_config('pytorch_learner_backend')
class PyTorchLearnerBackendConfig(BackendConfig):
    model: ModelConfig
    solver: SolverConfig
    log_tensorboard: bool = Field(
        True, description='If True, log events to Tensorboard log files.')
    run_tensorboard: bool = Field(
        False,
        description='If True, run Tensorboard server pointing at log files.')
    augmentors: List[str] = Field(
        default_augmentors,
        description=(
            'Names of albumentations augmentors to use for training batches. '
            'Choices include: ' + str(augmentor_list)))
    test_mode: bool = Field(
        False,
        description=
        ('This field is passed along to the LearnerConfig which is returned by '
         'get_learner_config(). For more info, see the docs for'
         'pytorch_learner.learner_config.LearnerConfig.test_mode.'))

    def get_bundle_filenames(self):
        return ['model-bundle.zip']

    def get_learner_config(self, pipeline):
        raise NotImplementedError()

    def build(self, pipeline, tmp_dir):
        raise NotImplementedError()
