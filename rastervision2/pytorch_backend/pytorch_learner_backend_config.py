from typing import List

from rastervision2.pipeline.config import register_config, Field
from rastervision2.core.backend import BackendConfig
from rastervision2.pytorch_learner.learner_config import (SolverConfig, ModelConfig,
                                                          default_augmentors,
                                                          augmentors as augmentor_list)


@register_config('pytorch_learner_backend')
class PyTorchLearnerBackendConfig(BackendConfig):
    model: ModelConfig
    solver: SolverConfig
    log_tensorboard: bool = Field(
        True, description='If True, log events to Tensorboard log files.')
    run_tensorboard: bool = Field(
        False, description='If True, run Tensorboard server pointing at log files.')
    augmentors: List[str] = Field(
        default_augmentors, description=(
            'Names of albumentations augmentors to use for training batches. '
            'Choices include: ' + str(augmentor_list)))

    def get_bundle_filenames(self):
        return ['model-bundle.zip']

    def get_learner_config(self, pipeline):
        raise NotImplementedError()

    def build(self, pipeline, tmp_dir):
        raise NotImplementedError()
