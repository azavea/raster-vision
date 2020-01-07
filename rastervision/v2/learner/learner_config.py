from typing import List

from rastervision.v2.core.pipeline_config import PipelineConfig
from rastervision.v2.core.config import Config, register_config


@register_config('model')
class ModelConfig(Config):
    backbone: str = 'resnet18'
    init_weights: str = None

    def update(self, learner=None):
        pass


@register_config('solver')
class SolverConfig(Config):
    lr: float = 1e-4
    num_epochs: int = 10
    test_num_epochs: int = 2
    overfit_num_steps: int = 1
    sync_interval: int = 5
    batch_sz: int = 32
    one_cycle: bool = True
    multi_stage: List = []

    def update(self, learner=None):
        pass


@register_config('data')
class DataConfig(Config):
    uri: str = None
    data_format: str = None
    labels: List[str] = []
    colors: List[str] = []
    img_sz: int = 256
    num_workers: int = 4

    def update(self, learner=None):
        pass


@register_config('learner')
class LearnerConfig(Config):
    model: ModelConfig
    solver: SolverConfig
    data: DataConfig

    predict_mode: bool = False
    test_mode: bool = False
    overfit_mode: bool = False
    eval_train: bool = False
    save_model_bundle: bool = True

    output_uri: str = None

    def update(self):
        super().update()

        if self.overfit_mode:
            self.data.img_sz = self.data.img_sz // 2
            if self.test_mode:
                self.solver.overfit_num_steps = self.solver.test_overfit_num_steps

        if self.test_mode:
            self.solver.num_epochs = self.solver.test_num_epochs
            self.data.img_sz = self.data.img_sz // 2
            self.solver.batch_sz = 4
            self.data.num_workers = 0

        self.model.update(learner=self)
        self.solver.update(learner=self)
        self.data.update(learner=self)

    def get_learner():
        from rastervision.v2.learner.learner import Learner
        return Learner


@register_config('learner_pipeline')
class LearnerPipelineConfig(PipelineConfig):
    learner: LearnerConfig

    def update(self):
        super().update()

        if self.learner.output_uri is None:
            self.learner.output_uri = self.root_uri

        self.learner.update()

    def get_pipeline(self):
        from rastervision.v2.learner.learner_pipeline import LearnerPipeline
        return LearnerPipeline
