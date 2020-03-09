from typing import List

from rastervision2.pipeline.config import register_config
from rastervision2.core.backend import BackendConfig
from rastervision2.pytorch_learner.object_detection_learner_config import (
    ObjectDetectionModelConfig, ObjectDetectionLearnerConfig,
    ObjectDetectionDataConfig)
from rastervision2.pytorch_learner.learner_config import (SolverConfig,
                                                          default_augmentors)
from rastervision2.pytorch_backend.pytorch_object_detection import (
    PyTorchObjectDetection)


@register_config('pytorch_object_detection_backend')
class PyTorchObjectDetectionConfig(BackendConfig):
    model: ObjectDetectionModelConfig
    solver: SolverConfig
    log_tensorboard: bool = True
    run_tensorboard: bool = False
    augmentors: List[str] = default_augmentors

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
            test_mode=pipeline.debug,
            output_uri=pipeline.train_uri,
            log_tensorboard=self.log_tensorboard,
            run_tensorboard=self.run_tensorboard)
        learner.update()
        return learner

    def build(self, pipeline, tmp_dir):
        learner = self.get_learner_config(pipeline)
        return PyTorchObjectDetection(pipeline, learner, tmp_dir)

    def get_bundle_filenames(self):
        return ['model-bundle.zip']
