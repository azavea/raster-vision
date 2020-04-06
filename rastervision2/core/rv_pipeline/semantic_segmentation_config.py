from typing import (List, Optional)

from rastervision2.pipeline.config import register_config, Config, Field
from rastervision2.core.rv_pipeline import RVPipelineConfig
from rastervision2.core.data import SemanticSegmentationLabelStoreConfig
from rastervision2.core.evaluation import SemanticSegmentationEvaluatorConfig

window_methods = ['sliding', 'random_sample']


@register_config('semantic_segmentation_chip_options')
class SemanticSegmentationChipOptions(Config):
    """Chipping options for semantic segmentation."""
    window_method: str = Field(
        'sliding',
        description=
        ('Window method to use for chipping. Options are: random_sample, sliding.'
         ))
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

    def validate_config(self):
        self.validate_list('window_method', window_methods)


@register_config('semantic_segmentation')
class SemanticSegmentationConfig(RVPipelineConfig):
    chip_options: SemanticSegmentationChipOptions = SemanticSegmentationChipOptions(
    )

    def build(self, tmp_dir):
        from rastervision2.core.rv_pipeline.semantic_segmentation import (
            SemanticSegmentation)
        return SemanticSegmentation(self, tmp_dir)

    def get_default_label_store(self, scene):
        return SemanticSegmentationLabelStoreConfig()

    def get_default_evaluator(self):
        return SemanticSegmentationEvaluatorConfig()

    def update(self):
        super().update()

        self.dataset.class_config.ensure_null_class()
