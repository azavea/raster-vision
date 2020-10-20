from typing import List, Optional
from pydantic.dataclasses import dataclass

from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator, ConfigError)
from rastervision.core.data.raster_transformer import RasterTransformerConfig
from rastervision.core.utils.misc import Proportion


@dataclass
class CropOffsets:
    """Tuple of relative offsets.

    Args:
        skip_top (Proportion): Proportion of height to exclude from the top.
        skip_left (Proportion): Proportion of width to exclude from the left.
        skip_bottom (Proportion): Proportion of height to exclude from the
            bottom.
        skip_right (Proportion): Proportion of width to exclude from the right.
    """
    skip_top: Proportion = 0.
    skip_left: Proportion = 0.
    skip_bottom: Proportion = 0.
    skip_right: Proportion = 0.

    def __iter__(self):
        return iter((self.skip_top, self.skip_left, self.skip_bottom,
                     self.skip_right))


@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: Optional[List[int]] = Field(
        None,
        description=
        'The sequence of channel indices to use when reading imagery.')
    transformers: List[RasterTransformerConfig] = []
    extent_crop: CropOffsets = Field(
        None,
        description='Relative offsets '
        '(skip_top, skip_left, skip_bottom, skip_right) for cropping '
        'the extent of the raster source. Useful for splitting a scene into '
        'different dataset splits. E.g. if you want to use the top 80% of the '
        'image for training and the bottom 20% for validation you can pass '
        'extent_crop=CropOffsets(skip_bottom=0.20) to the raster source in '
        'the training scene and extent_crop=CropOffsets(skip_top=0.80) to the '
        'raster source in the validation scene. Defaults to None i.e. no '
        'cropping.')

    def build(self, tmp_dir, use_transformers=True):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)

    @validator('extent_crop')
    def validate_extent_crop(cls, v):
        if v is None:
            return v
        skip_top, skip_left, skip_bottom, skip_right = v
        if skip_top + skip_bottom >= 1:
            raise ConfigError(
                'Invalid crop. skip_top + skip_bottom must be less than 1.')
        if skip_left + skip_right >= 1:
            raise ConfigError(
                'Invalid crop. skip_left + skip_right must be less than 1.')
        return v
