from typing import List, Optional, Tuple

from rastervision.pipeline.config import Config, register_config, Field
from rastervision.core.data.raster_transformer import RasterTransformerConfig


@register_config('raster_source')
class RasterSourceConfig(Config):
    channel_order: Optional[List[int]] = Field(
        None,
        description=
        'The sequence of channel indices to use when reading imagery.')
    transformers: List[RasterTransformerConfig] = []
    extent_crop: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description='Relative offsets (top, left, bottom, right) for cropping '
        'the extent of the raster source. Useful for using splitting a scene '
        'into different datasets. E.g. (0, 0, .8, 0) for the training set and '
        '(.8, 0, 0, 0) for the validation set will do a 80-20 split by '
        'height. Defaults to None i.e. no cropping.')

    def build(self, tmp_dir, use_transformers=True):
        raise NotImplementedError()

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
