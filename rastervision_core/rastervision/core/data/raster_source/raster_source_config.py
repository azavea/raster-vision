from typing import TYPE_CHECKING, List, Optional, Tuple

from rastervision.core.box import Box
from rastervision.pipeline.config import (Config, register_config, Field,
                                          validator, ConfigError)
from rastervision.core.data.raster_transformer import RasterTransformerConfig

if TYPE_CHECKING:
    from rastervision.core.data import (RasterSource, SceneConfig)
    from rastervision.core.rv_pipeline import RVPipelineConfig


def rs_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 6:
        # removed in version 7
        if cfg_dict.get('extent_crop') is not None:
            raise ConfigError('RasterSourceConfig.extent_crop is deprecated.')
        try:
            del cfg_dict['extent_crop']
        except KeyError:
            pass
    elif version == 9:
        # renamed in version 10
        cfg_dict['bbox'] = cfg_dict.get('extent')
        try:
            del cfg_dict['extent']
        except KeyError:
            pass
    return cfg_dict


@register_config('raster_source', upgrader=rs_config_upgrader)
class RasterSourceConfig(Config):
    """Configure a :class:`.RasterSource`."""

    channel_order: Optional[List[int]] = Field(
        None,
        description=
        'The sequence of channel indices to use when reading imagery.')
    transformers: List[RasterTransformerConfig] = []
    bbox: Optional[Tuple[int, int, int, int]] = Field(
        None,
        description='User-specified bbox in pixel coords in the form '
        '(ymin, xmin, ymax, xmax). Useful for cropping the raster source so '
        'that only part of the raster is read from.')

    def build(self,
              tmp_dir: Optional[str] = None,
              use_transformers: bool = True) -> 'RasterSource':
        raise NotImplementedError()

    def update(self,
               pipeline: Optional['RVPipelineConfig'] = None,
               scene: Optional['SceneConfig'] = None) -> None:
        for t in self.transformers:
            t.update(pipeline, scene)

    @validator('bbox')
    def validate_bbox(cls, v):
        if v is None:
            return None
        return Box(*v)
