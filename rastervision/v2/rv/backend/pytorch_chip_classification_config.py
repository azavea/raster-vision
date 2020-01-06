from rastervision.v2.rv.backend import BackendConfig
from rastervision.v2.learner.classification_config import (
    ClassificationLearnerConfig)
from rastervision.v2.core.config import register_config

@register_config('pytorch_chip_classification')
class PyTorchChipClassificationConfig(BackendConfig):
    learner: ClassificationLearnerConfig
