from typing import (List, Optional)

from rastervision2.pipeline.config import register_config, Config
from rastervision2.core.pipeline import RVPipelineConfig
from rastervision2.core.data import SemanticSegmentationLabelStoreConfig
from rastervision2.core.evaluation import SemanticSegmentationEvaluatorConfig

window_methods = ['sliding', 'random_sample']


@register_config('semantic_segmentation_chip_options')
class SemanticSegmentationChipOptions(Config):
    """
    Chipping options for semantic segmentation.

    Args:
        window_method: Window method to use for chipping. Options are:
            random_sample, sliding
        target_class_ids: list of class ids considered as targets (ie. those
            to prioritize when creating chips) which is only used in
            conjunction with the target_count_threshold and
            negative_survival_probability options.  Applies to the
            'random_sample' window method.
        negative_survival_prob: probability that a sampled negative
            chip will be utilized if it does not contain more pixels than
            target_count_threshold. Applies to the 'random_sample' window method.
        chips_per_scene: number of chips to generate per scene. Applies to
            the 'random_sample' window method.
        target_count_threshold: minimum number of pixels covering
            target_classes that a chip must have. Applies to the
            'random_sample' window method.
        stride: Stride of windows across image. Defaults to half the chip
            size. Applies to the 'sliding_window' method.
    """
    window_method: str = 'sliding'
    target_class_ids: Optional[List[int]] = None
    negative_survival_prob: float = 1.0
    chips_per_scene: int = 1000
    target_count_threshold: int = 1000
    stride: Optional[int] = None

    def validate_config(self):
        self.validate_list('window_method', window_methods)


@register_config('semantic_segmentation')
class SemanticSegmentationConfig(RVPipelineConfig):
    chip_options: SemanticSegmentationChipOptions = SemanticSegmentationChipOptions(
    )

    def build(self, tmp_dir):
        from rastervision2.core.pipeline.semantic_segmentation import SemanticSegmentation
        return SemanticSegmentation(self, tmp_dir)

    def get_default_label_store(self, scene):
        return SemanticSegmentationLabelStoreConfig()

    def get_default_evaluator(self):
        return SemanticSegmentationEvaluatorConfig()

    def update(self):
        super().update()

        self.dataset.class_config.ensure_null_class()
