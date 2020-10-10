from rastervision.pipeline.config import register_config
from rastervision.pytorch_backend.pytorch_learner_backend_config import (
    PyTorchLearnerBackendConfig)
from rastervision.pytorch_learner.semantic_segmentation_learner_config import (
    SemanticSegmentationModelConfig, SemanticSegmentationLearnerConfig,
    SemanticSegmentationDataConfig)
from rastervision.pytorch_backend.pytorch_semantic_segmentation import (
    PyTorchSemanticSegmentation)


@register_config('pytorch_semantic_segmentation_backend')
class PyTorchSemanticSegmentationConfig(PyTorchLearnerBackendConfig):
    model: SemanticSegmentationModelConfig

    def get_learner_config(self, pipeline):
        if self.img_sz is None:
            self.img_sz = pipeline.train_chip_sz

        data = SemanticSegmentationDataConfig(
            uri=pipeline.chip_uri,
            class_names=pipeline.dataset.class_config.names,
            class_colors=pipeline.dataset.class_config.colors,
            img_sz=self.img_sz,
            img_channels=pipeline.dataset.img_channels,
            img_format=pipeline.img_format,
            label_format=pipeline.label_format,
            num_workers=self.num_workers,
            augmentors=self.augmentors,
            base_transform=self.base_transform,
            aug_transform=self.aug_transform,
            plot_options=self.plot_options,
            channel_display_groups=pipeline.channel_display_groups)

        learner = SemanticSegmentationLearnerConfig(
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
        return PyTorchSemanticSegmentation(pipeline, learner, tmp_dir)
