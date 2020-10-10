from rastervision.pipeline.config import register_config
from rastervision.pytorch_backend.pytorch_learner_backend_config import (
    PyTorchLearnerBackendConfig)
from rastervision.pytorch_learner.classification_learner_config import (
    ClassificationModelConfig, ClassificationLearnerConfig,
    ClassificationDataConfig)

from rastervision.pytorch_backend.pytorch_chip_classification import (
    PyTorchChipClassification)


@register_config('pytorch_chip_classification_backend')
class PyTorchChipClassificationConfig(PyTorchLearnerBackendConfig):
    model: ClassificationModelConfig

    def get_learner_config(self, pipeline):
        if self.img_sz is None:
            self.img_sz = pipeline.train_chip_sz

        data = ClassificationDataConfig()
        data.uri = pipeline.chip_uri
        data.class_names = pipeline.dataset.class_config.names
        data.class_colors = pipeline.dataset.class_config.colors
        data.img_sz = self.img_sz
        data.augmentors = self.augmentors
        data.base_transform = self.base_transform
        data.aug_transform = self.aug_transform
        data.plot_options = self.plot_options
        data.num_workers = self.num_workers

        learner = ClassificationLearnerConfig(
            data=data,
            model=self.model,
            solver=self.solver,
            test_mode=self.test_mode,
            output_uri=pipeline.train_uri,
            log_tensorboard=self.log_tensorboard,
            run_tensorboard=self.run_tensorboard,
            predict_normalize=self.predict_normalize)
        learner.update()
        learner.validate_config()
        return learner

    def build(self, pipeline, tmp_dir):
        learner = self.get_learner_config(pipeline)
        return PyTorchChipClassification(pipeline, learner, tmp_dir)
