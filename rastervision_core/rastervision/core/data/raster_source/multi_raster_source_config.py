from pydantic import conint, conlist

from rastervision.pipeline.config import (Field, register_config, validator)
from rastervision.core.data.raster_source import (RasterSourceConfig,
                                                  MultiRasterSource)


def multi_rs_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 1:
        # field renamed in version 2
        cfg_dict['primary_source_idx'] = cfg_dict.get('crs_source', 0)
        try:
            del cfg_dict['crs_source']
        except KeyError:
            pass
    return cfg_dict


@register_config('multi_raster_source', upgrader=multi_rs_config_upgrader)
class MultiRasterSourceConfig(RasterSourceConfig):
    """Configure a :class:`.MultiRasterSource`."""

    raster_sources: conlist(
        RasterSourceConfig, min_items=1) = Field(
            ..., description='List of RasterSourceConfig to combine.')
    primary_source_idx: conint(ge=0) = Field(
        0,
        description=
        'Index of the raster source whose CRS, dtype, and other attributes '
        'will override those of the other raster sources. Defaults to 0.')
    force_same_dtype: bool = Field(
        False,
        description='Force all subchips to be of the same dtype as the '
        'primary_source_idx-th subchip.')

    @validator('primary_source_idx')
    def validate_primary_source_idx(cls, v: int, values: dict):
        raster_sources = values.get('raster_sources', [])
        if not (0 <= v < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')
        return v

    def build(self, tmp_dir: str,
              use_transformers: bool = True) -> MultiRasterSource:
        if use_transformers:
            raster_transformers = [t.build() for t in self.transformers]
        else:
            raster_transformers = []

        built_raster_sources = [
            rs.build(tmp_dir, use_transformers) for rs in self.raster_sources
        ]
        multi_raster_source = MultiRasterSource(
            raster_sources=built_raster_sources,
            primary_source_idx=self.primary_source_idx,
            force_same_dtype=self.force_same_dtype,
            channel_order=self.channel_order,
            raster_transformers=raster_transformers,
            extent=self.extent)
        return multi_raster_source

    def update(self, pipeline=None, scene=None):
        for t in self.transformers:
            t.update(pipeline, scene)
