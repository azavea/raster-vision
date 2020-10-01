from rastervision.pipeline.config import (register_config, validator,
                                          ConfigError)
from rastervision.pytorch_backend.pytorch_learner_backend_config import (
    PyTorchLearnerBackendConfig)
from rastervision.pytorch_learner.object_detection_learner_config import (
    ObjectDetectionModelConfig, ObjectDetectionLearnerConfig,
    ObjectDetectionDataConfig)
from rastervision.pytorch_backend.pytorch_object_detection import (
    PyTorchObjectDetection)


@register_config('pytorch_object_detection_backend')
class PyTorchObjectDetectionConfig(PyTorchLearnerBackendConfig):
    model: ObjectDetectionModelConfig

    def get_learner_config(self, pipeline):
        data = ObjectDetectionDataConfig()
        data.uri = pipeline.chip_uri
        data.class_names = pipeline.dataset.class_config.names
        data.class_colors = pipeline.dataset.class_config.colors
        data.img_sz = pipeline.train_chip_sz
        data.augmentors = self.augmentors
        data.base_transform = self.base_transform
        data.aug_transform = self.aug_transform
        data.plot_options = self.plot_options
        data.num_workers = self.num_workers

        learner = ObjectDetectionLearnerConfig(
            data=data,
            model=self.model,
            solver=self.solver,
            test_mode=self.test_mode,
            output_uri=pipeline.train_uri,
            log_tensorboard=self.log_tensorboard,
            run_tensorboard=self.run_tensorboard)
        learner.update()
        return learner

    def build(self, pipeline, tmp_dir):
        learner = self.get_learner_config(pipeline)
        return PyTorchObjectDetection(pipeline, learner, tmp_dir)

    @validator('model')
    def validate_model_config(cls, v):
        if v.external_def is not None:
            raise ConfigError('external_def is currently not supported for '
                              'Object Detection.')
        return v

    @validator('solver')
    def validate_solver_config(cls, v):
        if v.ignore_last_class:
            raise ConfigError(
                'ignore_last_class is not supported for Object Detection.')
        if v.class_loss_weights is not None:
            raise ConfigError(
                'class_loss_weights is currently not supported for '
                'Object Detection.')
        if v.external_loss_def is not None:
            raise ConfigError(
                'external_loss_def is currently not supported for '
                'Object Detection.')
        return v
