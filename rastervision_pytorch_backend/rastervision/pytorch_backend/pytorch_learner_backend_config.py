from typing import Optional

from rastervision.pipeline.config import (register_config, Field)
from rastervision.core.backend import BackendConfig
from rastervision.core.rv_pipeline import RVPipeline
from rastervision.pytorch_learner.learner_config import (
    SolverConfig, ModelConfig, DataConfig, ImageDataConfig)


@register_config('pytorch_learner_backend')
class PyTorchLearnerBackendConfig(BackendConfig):
    model: ModelConfig
    solver: SolverConfig
    data: DataConfig
    log_tensorboard: bool = Field(
        True, description='If True, log events to Tensorboard log files.')
    run_tensorboard: bool = Field(
        False,
        description='If True, run Tensorboard server pointing at log files.')
    test_mode: bool = Field(
        False,
        description=
        ('This field is passed along to the LearnerConfig which is returned by '
         'get_learner_config(). For more info, see the docs for'
         'pytorch_learner.learner_config.LearnerConfig.test_mode.'))

    def get_bundle_filenames(self):
        return ['model-bundle.zip']

    def update(self, pipeline: Optional[RVPipeline] = None):
        super().update(pipeline=pipeline)

        if isinstance(self.data, ImageDataConfig):
            if self.data.uri is None and self.data.group_uris is None:
                self.data.uri = pipeline.chip_uri
        if not self.data.class_names:
            self.data.class_names = pipeline.dataset.class_config.names
        if not self.data.class_colors:
            self.data.class_colors = pipeline.dataset.class_config.colors

    def get_learner_config(self, pipeline: Optional[RVPipeline]):
        raise NotImplementedError()

    def build(self, pipeline: Optional[RVPipeline], tmp_dir: str):
        raise NotImplementedError()
