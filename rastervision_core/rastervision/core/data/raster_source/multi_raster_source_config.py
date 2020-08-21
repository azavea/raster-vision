from typing import List, Optional, Tuple, Union

import numpy as np

from rastervision.pipeline.config import (Config, register_config, Field,
                                          ConfigError)
from rastervision.core.data.raster_transformer import RasterTransformerConfig
from rastervision.core.data.raster_source import RasterSourceConfig, RasterioSourceConfig, MultiRasterSource


@register_config('multi_raster_source')
class MultiRasterSourceConfig(RasterSourceConfig):
    raster_sources: List[Tuple[RasterSourceConfig, Union[tuple, List[
        int]]]] = Field(
            ...,
            description='List of (RasterSourceConfig, indices) pair, where '
            'are the positions to which each channel of the raster source will '
            'be sent.')

    def get_raw_channel_order(self):
        channel_mappings = sum((list(inds) for _, inds in self.raster_sources),
                               [])
        self.validate_channel_mappings(channel_mappings)
        raw_channel_order = np.argsort(channel_mappings)

        return raw_channel_order

    def validate_channel_mappings(self, channel_mappings):
        # ensure we have a source channel for each channel idx
        if set(channel_mappings) != set(range(len(channel_mappings))):
            raise ConfigError(f'Missing mappings for some channels.')

    def validate_config(self):
        super().validate_config()

        if len(self.raster_sources) == 0:
            raise ConfigError(f'No raster sources provided.')

        for i, (_, inds) in enumerate(self.raster_sources):
            if len(inds) == 0:
                raise ConfigError(
                    f'Got no indices for raster source at index {i}.')
            if isinstance(inds, tuple):
                if any(not isinstance(ind, int) for ind in inds):
                    raise ConfigError(f'Indices must be ints. Got: {ind}.')
            if any(ind < 0 for ind in inds):
                raise ConfigError(f'Indices must be >= 0. Got: {ind}.')

    def build(self, tmp_dir, use_transformers=True):
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

        raster_sources = [
            cfg.build(tmp_dir, use_transformers) for cfg, _ in self.raster_sources
        ]
        multi_raster_source = MultiRasterSource(
            raster_sources=raster_sources,
            raw_channel_order=self.get_raw_channel_order(),
            raster_transformers=raster_transformers)
        return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
