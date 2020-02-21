from typing import List

from rastervision2.pipeline.config import register_config
from rastervision2.core.backend import BackendConfig
from rastervision2.pytorch_learner.semantic_segmentation_config import (
    SemanticSegmentationModelConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationDataConfig)
from rastervision2.pytorch_learner.learner_config import (SolverConfig,
                                                          default_augmentors)
from rastervision2.pytorch_backend.pytorch_semantic_segmentation import (
    PyTorchSemanticSegmentation)


@register_config('pytorch_semantic_segmentation_backend')
class PyTorchSemanticSegmentationConfig(BackendConfig):
    model: SemanticSegmentationModelConfig
    solver: SolverConfig
    log_tensorboard: bool = True
    run_tensorboard: bool = False
    augmentors: List[str] = default_augmentors

    def get_learner_config(self, pipeline):
        data = SemanticSegmentationDataConfig()
        data.uri = pipeline.chip_uri
        data.class_names = pipeline.dataset.class_config.names
        data.class_colors = pipeline.dataset.class_config.colors
        data.img_sz = pipeline.train_chip_sz
        data.augmentors = self.augmentors

        learner = SemanticSegmentationLearnerConfig(
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
        return PyTorchSemanticSegmentation(pipeline, learner, tmp_dir)

    def get_bundle_filenames(self):
        return ['model-bundle.zip']
