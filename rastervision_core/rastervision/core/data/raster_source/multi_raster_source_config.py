from typing import Sequence
from pydantic import conint

import numpy as np

from rastervision.pipeline.config import (Config, register_config, Field,
                                          ConfigError, validator)
from rastervision.core.data.raster_transformer import RasterTransformerConfig
from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  MultiRasterSource)


@register_config('sub_raster_source')
class SubRasterSourceConfig(Config):
    raster_source: RasterSourceConfig = Field(
        ...,
        description=
        'A RasterSourceConfig that will provide some of the channels.')
    target_channels: Sequence[conint(ge=0)] = Field(
        ...,
        description='Channel indices to send each of the channels in this '
        'raster source to.')

    @validator('target_channels')
    def non_empty_target_channels(cls, v):
        if len(v) == 0:
            raise ConfigError('target_channels should be non-empty.')
        return list(v)


@register_config('multi_raster_source')
class MultiRasterSourceConfig(RasterSourceConfig):
    raster_sources: Sequence[SubRasterSourceConfig] = Field(
        ..., description='List of SubRasterSourceConfig to combine.')

    def get_raw_channel_order(self):
        channel_mappings = sum(
            (rs.target_channels for rs in self.raster_sources), [])
        self.validate_channel_mappings(channel_mappings)
        raw_channel_order = np.argsort(channel_mappings)

        return raw_channel_order

    def validate_channel_mappings(self, channel_mappings: Sequence[int]):
        src_inds = set(range(len(channel_mappings)))
        tgt_inds = set(channel_mappings)
        if src_inds != tgt_inds:
            raise ConfigError(f'Missing mappings for some channels.')

    @validator('raster_sources')
    def validate_raster_sources(cls, v):
        if len(v) == 0:
            raise ConfigError('raster_sources should be non-empty.')
        return v

    def build(self, tmp_dir, use_transformers=True):
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

        built_raster_sources = [
            sub_rs.raster_source.build(tmp_dir, use_transformers)
            for sub_rs in self.raster_sources
        ]
        multi_raster_source = MultiRasterSource(
            raster_sources=built_raster_sources,
            raw_channel_order=self.get_raw_channel_order(),
            raster_transformers=raster_transformers)
        return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
