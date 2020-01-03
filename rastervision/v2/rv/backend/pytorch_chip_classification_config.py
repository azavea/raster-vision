from rastervision.v2.rv.backend import BackendConfig
from rastervision.v2.learner.classification_config import (
    ClassificationLearnerConfig)

class PyTorchChipClassificationConfig(BackendConfig):
    learner: ClassificationLearnerConfig
