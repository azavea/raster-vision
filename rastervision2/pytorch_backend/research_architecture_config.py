from typing import Optional, List

from rastervision2.pipeline.config import register_config
from rastervision2.pytorch_backend.pytorch_learner_backend_config import (
    PyTorchLearnerBackendConfig)
from rastervision2.pytorch_learner.semantic_segmentation_learner_config import (
    SemanticSegmentationModelConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationDataConfig)
from rastervision2.pytorch_backend.pytorch_semantic_segmentation import (
    PyTorchSemanticSegmentation)
from rastervision2.core.backend import BackendConfig


@register_config('research_architecture_config')
class ResearchArchitectureConfig(BackendConfig):
    # model: SemanticSegmentationModelConfig

    # def get_learner_config(self, pipeline):
    #     data = SemanticSegmentationDataConfig()
    #     data.uri = pipeline.chip_uri
    #     data.class_names = pipeline.dataset.class_config.names
    #     data.class_colors = pipeline.dataset.class_config.colors
    #     data.img_sz = pipeline.train_chip_sz
    #     data.augmentors = self.augmentors

    #     learner = SemanticSegmentationLearnerConfig(
    #         data=data,
    #         model=self.model,
    #         solver=self.solver,
    #         test_mode=self.test_mode,
    #         output_uri=pipeline.train_uri,
    #         log_tensorboard=self.log_tensorboard,
    #         run_tensorboard=self.run_tensorboard)
    #     learner.update()
    #     return learner

    # def build(self, pipeline, tmp_dir):
    #     learner = self.get_learner_config(pipeline)
    #     return PyTorchSemanticSegmentation(pipeline, learner, tmp_dir)
    def build(self, pipeline: 'RVPipeline', tmp_dir: str) -> 'Backend':
        raise NotImplementedError()

    def get_bundle_filenames(self) -> List[str]:
        raise NotImplementedError()

    def update(self, pipeline: Optional['RVPipeline'] = None):  # noqa
        pass
