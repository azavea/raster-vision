from enum import Enum
from typing import Optional

from rastervision2.pipeline.config import Field, register_config, validator
from rastervision2.pytorch_learner.learner_config import (Backbone, DataConfig,
                                                          LearnerConfig,
                                                          ModelConfig)


@register_config('research_architecture_data')
class ResearchArchitectureDataConfig(DataConfig):
    pass

@register_config('research_architecture_learner')
class ResearchArchitectureLearnerConfig(LearnerConfig):
    data: ResearchArchitectureDataConfig
    model: Optional[ModelConfig] = None

    def build(self, tmp_dir, model_path=None):
        from rastervision2.pytorch_learner.research_architecture_learner import (ResearchArchitectureLearner)
        return ResearchArchitectureLearner(
            self, tmp_dir, model_path=model_path)
