from typing import (List, Optional)
from enum import Enum

from rastervision.pipeline.config import register_config, Config, Field
from rastervision.core.rv_pipeline import RVPipelineConfig
from rastervision.core.data import SemanticSegmentationLabelStoreConfig
from rastervision.core.evaluation import SemanticSegmentationEvaluatorConfig


class SemanticSegmentationWindowMethod(Enum):
    """Enum for window methods

    Attributes:
        sliding: use a sliding window
        random_sample: randomly sample windows
    """

    sliding = 'sliding'
    random_sample = 'random_sample'


@register_config('semantic_segmentation_chip_options')
class SemanticSegmentationChipOptions(Config):
    """Chipping options for semantic segmentation."""
    window_method: SemanticSegmentationWindowMethod = Field(
        SemanticSegmentationWindowMethod.sliding,
        description=('Window method to use for chipping.'))
    target_class_ids: Optional[List[int]] = Field(
        None,
        description=
        ('List of class ids considered as targets (ie. those to prioritize when '
         'creating chips) which is only used in conjunction with the '
         'target_count_threshold and negative_survival_probability options. Applies '
         'to the random_sample window method.'))
    negative_survival_prob: float = Field(
        1.0,
        description=
        ('List of class ids considered as targets (ie. those to prioritize when creating '
         'chips) which is only used in conjunction with the target_count_threshold and '
         'negative_survival_probability options. Applies to the random_sample window '
         'method.'))
    chips_per_scene: int = Field(
        1000,
        description=
        ('Number of chips to generate per scene. Applies to the random_sample window '
         'method.'))
    target_count_threshold: int = Field(
        1000,
        description=
        ('Minimum number of pixels covering target_classes that a chip must have. '
         'Applies to the random_sample window method.'))
    stride: Optional[int] = Field(
        None,
        description=
        ('Stride of windows across image. Defaults to half the chip size. Applies to '
         'the sliding_window method.'))


@register_config('semantic_segmentation')
class SemanticSegmentationConfig(RVPipelineConfig):
    chip_options: SemanticSegmentationChipOptions = SemanticSegmentationChipOptions(
    )

    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.semantic_segmentation import (
            SemanticSegmentation)
        return SemanticSegmentation(self, tmp_dir)

    def get_default_label_store(self, scene):
        return SemanticSegmentationLabelStoreConfig()

    def get_default_evaluator(self):
        return SemanticSegmentationEvaluatorConfig()

    def update(self):
        super().update()

        self.dataset.class_config.ensure_null_class()
