from typing import (List, Optional, Union)
from enum import Enum

from rastervision.pipeline.config import (register_config, Config, ConfigError,
                                          Field)
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


@register_config('semantic_segmentation_predict_options')
class SemanticSegmentationPredictOptions(PredictOptions):
    stride: Optional[int] = Field(
        None,
        description=
        'Stride of windows across image. Allows aggregating multiple '
        'predictions for each pixel if less than the chip size and outputting '
        'smooth labels. Defaults to predict_chip_sz.')


@register_config('semantic_segmentation')
class SemanticSegmentationConfig(RVPipelineConfig):
    chip_options: SemanticSegmentationChipOptions = \
        SemanticSegmentationChipOptions()
    predict_options: SemanticSegmentationPredictOptions = \
        SemanticSegmentationPredictOptions()

    channel_display_groups: Optional[Union[dict, list, tuple]] = Field(
        None,
        description=
        ('Groups of image channels to display together as a subplot '
         'when plotting the data and predictions. '
         'Can be a list or tuple of groups (e.g. [(0, 1, 2), (3,)]) or a dict '
         'containing title-to-group mappings '
         '(e.g. {"RGB": [0, 1, 2], "IR": [3]}), '
         'where each group is a list or tuple of channel indices and title '
         'is a string that will be used as the title of the subplot '
         'for that group.'))

    img_format: Optional[str] = Field(
        None, description='The filetype of the training images.')
    label_format: str = Field(
        'png', description='The filetype of the training labels.')

    def build(self, tmp_dir):
        from rastervision.core.rv_pipeline.semantic_segmentation import (
            SemanticSegmentation)
        return SemanticSegmentation(self, tmp_dir)

    def update(self):
        super().update()

        self.dataset.class_config.ensure_null_class()

        if self.dataset.img_channels is None:
            return

        if self.img_format is None:
            self.img_format = 'png' if self.dataset.img_channels == 3 else 'npy'

        if self.channel_display_groups is None:
            img_channels = min(3, self.dataset.img_channels)
            self.channel_display_groups = {'Input': tuple(range(img_channels))}

    def validate_config(self):
        super().validate_config()

        if self.dataset.img_channels is None:
            return

        if self.img_format == 'png' and self.dataset.img_channels != 3:
            raise ConfigError('img_channels must be 3 if img_format is png.')

        self.validate_channel_display_groups()

    def get_default_label_store(self, scene):
        return SemanticSegmentationLabelStoreConfig()

    def get_default_evaluator(self):
        return SemanticSegmentationEvaluatorConfig()

    def validate_channel_display_groups(self):
        def _are_ints(ints) -> bool:
            return all(isinstance(i, int) for i in ints)

        def _in_range(inds, lt: int) -> bool:
            return all(0 <= i < lt for i in inds)

        img_channels = self.dataset.img_channels
        groups = self.channel_display_groups

        # validate dict form
        if isinstance(groups, dict):
            for k, v in groups.items():
                if not isinstance(k, str):
                    raise ConfigError(
                        'channel_display_groups keys must be strings.')
                if not isinstance(v, (list, tuple)):
                    raise ConfigError(
                        'channel_display_groups values must be lists or tuples.'
                    )
                if not (0 < len(v) <= 3):
                    raise ConfigError(
                        f'channel_display_groups[{k}]: len(group) must be 1, 2, or 3'
                    )
                if not (_are_ints(v) and _in_range(v, lt=img_channels)):
                    raise ConfigError(
                        f'Invalid channel indices in channel_display_groups[{k}].'
                    )
        # validate list/tuple form
        elif isinstance(groups, (list, tuple)):
            for i, grp in enumerate(groups):
                if not (0 < len(grp) <= 3):
                    raise ConfigError(
                        f'channel_display_groups[{i}]: len(group) must be 1, 2, or 3'
                    )
                if not (_are_ints(grp) and _in_range(grp, lt=img_channels)):
                    raise ConfigError(
                        f'Invalid channel index in channel_display_groups[{i}].'
                    )
