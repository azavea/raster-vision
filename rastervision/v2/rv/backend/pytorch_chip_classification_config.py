from rastervision.v2.rv.backend import BackendConfig
from rastervision.v2.learner.classification_config import (
    ClassificationLearnerConfig)
from rastervision.v2.core.config import register_config
from rastervision.v2.rv.backend.pytorch_chip_classification import (
    PyTorchChipClassification)
from rastervision.v2.rv.task.chip_classification_config import (
    ChipClassificationConfig)

@register_config('pytorch_chip_classification')
class PyTorchChipClassificationConfig(BackendConfig):
    learner: ClassificationLearnerConfig

    def update(self, parent=None):
        if isinstance(parent, ChipClassificationConfig):
            parent: ChipClassificationConfig
            self.learner.data.img_sz = parent.train_chip_sz
            self.learner.test_mode = parent.debug
            self.learner.data.colors = parent.dataset.class_config.colors
            self.learner.data.labels = parent.dataset.class_config.names

        super().update(parent)

    def build(self, tmp_dir):
        learner = self.learner.get_learner()(self.learner, self.tmp_dir)
        learner = self.learner.build()
        return PyTorchChipClassification(learner)