from rastervision2.pipeline.config import register_config
from rastervision2.core.backend import BackendConfig
from rastervision2.pytorch_learner.classification_config import (
    ClassificationModelConfig, ClassificationLearnerConfig,
    ClassificationDataConfig)
from rastervision2.pytorch_learner.learner_config import (
    SolverConfig)
from rastervision2.pytorch_backend.pytorch_chip_classification import (
    PyTorchChipClassification)


@register_config('pytorch_chip_classification')
class PyTorchChipClassificationConfig(BackendConfig):
    model: ClassificationModelConfig
    solver: SolverConfig

    def get_learner_config(self, task):
        data = ClassificationDataConfig()
        data.uri = task.chip_uri
        data.class_names = task.dataset.class_config.names
        data.class_colors = task.dataset.class_config.colors
        data.img_sz = task.train_chip_sz

        learner = ClassificationLearnerConfig(
            data=data, model=self.model, solver=self.solver,
            test_mode=task.debug, output_uri=task.train_uri)
        learner.update()
        return learner

    def build(self, task, tmp_dir):
        learner = self.get_learner_config(task)
        return PyTorchChipClassification(task, learner, tmp_dir)

    def get_bundle_filenames(self):
        return ['model-bundle.zip']
