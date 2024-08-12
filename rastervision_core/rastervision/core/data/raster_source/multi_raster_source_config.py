from typing import TYPE_CHECKING

from typing_extensions import Annotated
from pydantic import NonNegativeInt as NonNegInt

from rastervision.pipeline.config import (Field, register_config,
                                          model_validator)
from rastervision.core.box import Box
from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  MultiRasterSource)

if TYPE_CHECKING:
    from typing import Self


def multi_rs_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 1:
        # field renamed in version 2
        cfg_dict['primary_source_idx'] = cfg_dict.get('crs_source', 0)
        cfg_dict.pop('crs_source', None)
    elif version == 13:
        # field removed in version 14
        cfg_dict.pop('force_same_dtype', None)
    return cfg_dict


@register_config('multi_raster_source', upgrader=multi_rs_config_upgrader)
class MultiRasterSourceConfig(RasterSourceConfig):
    """Configure a :class:`.MultiRasterSource`.

    Or :class:`.TemporalMultiRasterSource`, if ``temporal=True``.
    """

    raster_sources: Annotated[list[
        RasterSourceConfig], Field(min_length=1)] = Field(
            ..., description='List of RasterSourceConfig to combine.')
    primary_source_idx: NonNegInt = Field(
        0,
        description=
        'Index of the raster source whose CRS, dtype, and other attributes '
        'will override those of the other raster sources. Defaults to 0.')
    temporal: bool = Field(
        False,
        description='Stack images from sub raster sources into a time-series '
        'of shape (T, H, W, C) instead of concatenating bands.')

    @model_validator(mode='after')
    def validate_primary_source_idx(self) -> 'Self':
        primary_source_idx = self.primary_source_idx
        raster_sources = self.raster_sources
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')
        return self

    @model_validator(mode='after')
    def validate_temporal(self) -> 'Self':
        if self.temporal and self.channel_order is not None:
            raise ValueError(
                'Setting channel_order is not allowed if temporal=True.')
        return self

    def build(self, tmp_dir: str | None = None,
              use_transformers: bool = True) -> MultiRasterSource:
        if use_transformers:
            raster_transformers = [
                t.build(channel_order=self.channel_order)
                for t in self.transformers
            ]
        else:
            raster_transformers = []

        built_raster_sources = [
            rs.build(tmp_dir, use_transformers) for rs in self.raster_sources
        ]
        bbox = Box(*self.bbox) if self.bbox is not None else None

        if self.temporal:
            from rastervision.core.data.raster_source import (
                TemporalMultiRasterSource)
            multi_raster_source = TemporalMultiRasterSource(
                raster_sources=built_raster_sources,
                primary_source_idx=self.primary_source_idx,
                raster_transformers=raster_transformers,
                bbox=bbox)
        else:
            multi_raster_source = MultiRasterSource(
                raster_sources=built_raster_sources,
                primary_source_idx=self.primary_source_idx,
                channel_order=self.channel_order,
                raster_transformers=raster_transformers,
                bbox=bbox)
        return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
