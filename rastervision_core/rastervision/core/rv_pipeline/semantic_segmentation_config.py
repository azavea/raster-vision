from typing import TYPE_CHECKING, Literal
import logging

from pydantic import NonNegativeInt as NonNegInt
import numpy as np

from rastervision.pipeline.config import (register_config, Field,
                                          model_validator)
from rastervision.core.rv_pipeline.rv_pipeline_config import (PredictOptions,
                                                              RVPipelineConfig)
from rastervision.core.rv_pipeline.chip_options import (ChipOptions,
                                                        WindowSamplingConfig)
from rastervision.core.data import SemanticSegmentationLabelStoreConfig
from rastervision.core.evaluation import SemanticSegmentationEvaluatorConfig

if TYPE_CHECKING:
    from typing import Self

log = logging.getLogger(__name__)


def ss_chip_options_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 10:
        sampling = WindowSamplingConfig(
            method=cfg_dict.pop('window_method', None),
            size=300,
            stride=cfg_dict.pop('stride', None),
            max_windows=cfg_dict.pop('chips_per_scene', None),
        )
        cfg_dict['sampling'] = sampling.dict()
    return cfg_dict


@register_config(
    'semantic_segmentation_chip_options', upgrader=ss_chip_options_upgrader)
class SemanticSegmentationChipOptions(ChipOptions):
    """Chipping options for semantic segmentation."""
    target_class_ids: list[int] | None = Field(
        None,
        description=
        ('List of class ids considered as targets (ie. those to prioritize when '
         'creating chips) which is only used in conjunction with the '
         'target_count_threshold and negative_survival_probability options. Applies '
         'to the random_sample window method.'))
    negative_survival_prob: float = Field(
        1.0, description='Probability of keeping a negative chip.')
    target_count_threshold: int = Field(
        1000,
        description=
        ('Minimum number of pixels covering target_classes that a chip must have. '
         'Applies to the random_sample window method.'))

    def keep_chip(self, chip: np.ndarray, label: np.ndarray) -> bool:
        keep = super().keep_chip(chip, label)
        if not keep:
            return False
        if self.target_class_ids is not None:
            if self.enough_target_pixels(label):
                return True
            if np.random.sample() <= self.negative_survival_prob:
                return True
            return False
        return keep

    def enough_target_pixels(self, label_arr: np.ndarray) -> bool:
        """Check if label raster has enough pixels of the target classes.

        Args:
             label_arr: The label raster for a chip.

        Returns:
             True (the window does contain interesting pixels) or False.
        """
        target_count = 0
        if self.target_class_ids is None:
            raise ValueError('target_class_ids not specified.')
        for class_id in self.target_class_ids:
            target_count += (label_arr == class_id).sum()
        enough_target_pixels = target_count >= self.target_count_threshold
        return enough_target_pixels


@register_config('semantic_segmentation_predict_options')
class SemanticSegmentationPredictOptions(PredictOptions):
    stride: int | None = Field(
        None,
        description='Stride of the sliding window for generating chips. '
        'Allows aggregating multiple predictions for each pixel if less than '
        'the chip size. Defaults to ``chip_sz``.')
    crop_sz: NonNegInt | Literal['auto'] | None = Field(
        None,
        description=
        'Number of rows/columns of pixels from the edge of prediction '
        'windows to discard. This is useful because predictions near edges '
        'tend to be lower quality and can result in very visible artifacts '
        'near the edges of chips. If "auto", will be set to half the stride '
        'if stride is less than chip_sz. Defaults to None.')

    @model_validator(mode='after')
    def set_auto_crop_sz(self) -> 'Self':
        if self.crop_sz == 'auto':
            if self.stride is None:
                self.validate_stride()
            overlap_sz = self.chip_sz - self.stride
            if overlap_sz % 2 == 1:
                log.warning(
                    'Using crop_sz="auto" but overlap size (chip_sz minus '
                    'stride) is odd. This means that one pixel row/col will '
                    'still overlap after cropping.')
            self.crop_sz = overlap_sz // 2
        return self


def ss_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 0:
        try:
            # removed in version 1
            del cfg_dict['channel_display_groups']
            del cfg_dict['img_format']
            del cfg_dict['label_format']
        except KeyError:
            pass
    return cfg_dict


@register_config('semantic_segmentation', upgrader=ss_config_upgrader)
class SemanticSegmentationConfig(RVPipelineConfig):
    """Configure a :class:`.SemanticSegmentation` pipeline."""

    chip_options: SemanticSegmentationChipOptions | None = None
    predict_options: SemanticSegmentationPredictOptions | None = None

    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.semantic_segmentation import (
            SemanticSegmentation)
        return SemanticSegmentation(self, tmp_dir)

    def update(self):
        self.dataset.class_config.ensure_null_class()
        super().update()

    def validate_config(self):
        super().validate_config()

    def get_default_label_store(self, scene):
        return SemanticSegmentationLabelStoreConfig()

    def get_default_evaluator(self):
        return SemanticSegmentationEvaluatorConfig()
