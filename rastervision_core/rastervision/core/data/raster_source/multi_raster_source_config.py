from typing import Sequence
from pydantic import conint

from rastervision.pipeline.config import (Config, register_config, Field,
                                          ConfigError, validator)
from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  MultiRasterSource)


@register_config('sub_raster_source')
class SubRasterSourceConfig(Config):
    raster_source: RasterSourceConfig = Field(
        ...,
        description=
        'A RasterSourceConfig that will provide a subset of the channels.')
    target_channels: Sequence[conint(ge=0)] = Field(
        ...,
        description='Channel indices to send each of the channels in this '
        'raster source to.')

    @validator('target_channels')
    def non_empty_target_channels(cls, v):
        if len(v) == 0:
            raise ConfigError('target_channels should be non-empty.')
        return list(v)

    def build(self, tmp_dir, use_transformers=True):
        rs = self.raster_source.build(tmp_dir, use_transformers)
        return rs


@register_config('multi_raster_source')
class MultiRasterSourceConfig(RasterSourceConfig):
    raster_sources: Sequence[SubRasterSourceConfig] = Field(
        ..., description='List of SubRasterSourceConfigs to combine.')
    allow_different_extents: bool = Field(
        False, description='Allow sub-rasters to have different extents.')
    force_same_dtype: bool = Field(
        False,
        description=
        'Force all subchips to be of the same dtype as the first subchip.')
    crs_source: conint(ge=0) = Field(
        0,
        description=
        'Use the crs_transformer of the raster source at this index.')

    def get_raw_channel_order(self):
        # concatenate all target_channels
        channel_mappings = sum(
            (rs.target_channels for rs in self.raster_sources), [])

        # this will be used to index the channel dim of the
        # concatenated array to achieve the channel mappings
        raw_channel_order = [0] * len(channel_mappings)
        for from_idx, to_idx in enumerate(channel_mappings):
            raw_channel_order[to_idx] = from_idx

        self.validate_channel_mappings(channel_mappings, raw_channel_order)

        return raw_channel_order

    def validate_channel_mappings(self, channel_mappings: Sequence[int],
                                  raw_channel_order: Sequence[int]):
        # validate completeness of mappings
        src_inds = set(range(len(channel_mappings)))
        tgt_inds = set(channel_mappings)
        if src_inds != tgt_inds:
            raise ConfigError('Missing mappings for some channels.')

        # check compatibility with channel_order, if given
        if self.channel_order:
            if len(self.channel_order) != len(raw_channel_order):
                raise ConfigError(
                    f'Channel mappings ({raw_channel_order}) and '
                    f'channel_order ({self.channel_order}) are incompatible.')

    @validator('raster_sources')
    def validate_raster_sources(cls, v):
        if len(v) == 0:
            raise ConfigError('raster_sources should be non-empty.')
        return v

    def build(self, tmp_dir, use_transformers=True):
        if use_transformers:
            raster_transformers = [t.build() for t in self.transformers]
        else:
            raster_transformers = []

        built_raster_sources = [
            rs.build(tmp_dir, use_transformers) for rs in self.raster_sources
        ]
        multi_raster_source = MultiRasterSource(
            raster_sources=built_raster_sources,
            raw_channel_order=self.get_raw_channel_order(),
            force_same_dtype=self.force_same_dtype,
            allow_different_extents=self.allow_different_extents,
            channel_order=self.channel_order,
            crs_source=self.crs_source,
            raster_transformers=raster_transformers,
            extent_crop=self.extent_crop)
        return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
