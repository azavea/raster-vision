from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_learner.learner_config import (
    DataConfig, LearnerConfig, ModelConfig)


@register_config('research_architecture_learner')
class ResearchArchitectureLearnerConfig(LearnerConfig):
    architecture: str
    pretrained: bool
    bands: int
    resolution_divisor: int
    model: ModelConfig = ModelConfig()
    data: DataConfig

    def build(self, tmp_dir, model_path=None):
        from rastervision2.research_backend.research_architecture_learner import (
            ResearchArchitectureLearner)
        return ResearchArchitectureLearner(
            self, tmp_dir, model_path=model_path)
