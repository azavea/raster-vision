from typing import (List, Optional, Union)
from typing_extensions import Literal
from pydantic import conint
from enum import Enum

from rastervision.pipeline.config import (ConfigError, register_config, Field,
                                          Config, validator)
from rastervision.core.rv_pipeline.rv_pipeline_config import (RVPipelineConfig,
                                                              PredictOptions)
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


def ss_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version < 1:
        try:
            # removed in version 1
            del cfg_dict['channel_display_groups']
            del cfg_dict['img_format']
            del cfg_dict['label_format']
        except KeyError:
            pass
    return cfg_dict


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
        1.0, description='Probability of keeping a negative chip.')
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


@register_config('semantic_segmentation_predict_options')
class SemanticSegmentationPredictOptions(PredictOptions):
    stride: Optional[int] = Field(
        None,
        description=
        'Stride of windows across image. Allows aggregating multiple '
        'predictions for each pixel if less than the chip size. '
        'Defaults to predict_chip_sz.')
    crop_sz: Optional[Union[conint(gt=0), Literal['auto']]] = Field(
        None,
        description=
        'Number of rows/columns of pixels from the edge of prediction '
        'windows to discard. This is useful because predictions near edges '
        'tend to be lower quality and can result in very visible artifacts '
        'near the edges of chips. If "auto", will be set to half the stride '
        'if stride is less than chip_sz. Defaults to None.')

    @validator('crop_sz')
    def validate_crop_sz(cls,
                         v: Optional[Union[conint(gt=0), Literal['auto']]],
                         values: dict) -> dict:
        stride: Optional[int] = values.get('stride')
        crop_sz = v

        if stride is None and crop_sz is not None:
            raise ConfigError('Cannot use crop_sz if stride is None.')

        return crop_sz


@register_config('semantic_segmentation', upgrader=ss_config_upgrader)
class SemanticSegmentationConfig(RVPipelineConfig):
    """Configure a :class:`.SemanticSegmentation` pipeline."""

    chip_options: SemanticSegmentationChipOptions = \
        SemanticSegmentationChipOptions()
    predict_options: SemanticSegmentationPredictOptions = \
        SemanticSegmentationPredictOptions()

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
