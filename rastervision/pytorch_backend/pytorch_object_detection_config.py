from rastervision.pipeline.config import register_config
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
