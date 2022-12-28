from typing import List, Union

from rastervision.core.data.raster_source import RasterSourceConfig, RasterioSource
from rastervision.pipeline.config import ConfigError, Field, register_config


def rasterio_source_config_upgrader(cfg_dict: dict, version: int) -> dict:
    if version == 5:
        # removed in version 6
        x_shift = cfg_dict.get('x_shift', 0)
        y_shift = cfg_dict.get('y_shift', 0)
        if x_shift != 0 or y_shift != 0:
            raise ConfigError('x_shift and y_shift are deprecated. '
                              'Use the ShiftTrasnformer instead.')
        try:
            del cfg_dict['x_shift']
            del cfg_dict['y_shift']
        except KeyError:
            pass
    return cfg_dict


@register_config('rasterio_source', upgrader=rasterio_source_config_upgrader)
class RasterioSourceConfig(RasterSourceConfig):
    """Configure a :class:`.RasterioSource`."""

    uris: Union[str, List[str]] = Field(
        ...,
        description='One or more image URIs that comprise the imagery for a '
        'scene. The format of each file can be any that can be read by '
        'Rasterio/GDAL. If > 1 URI is provided, a VRT will be created to '
        'mosaic together the individual images.')
    allow_streaming: bool = Field(
        False,
        description='Stream assets as needed rather than downloading them.')

    def build(self, tmp_dir, use_transformers=True):
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

        return RasterioSource(
            uris=self.uris,
            raster_transformers=raster_transformers,
            tmp_dir=tmp_dir,
            allow_streaming=self.allow_streaming,
            channel_order=self.channel_order,
            extent=self.extent)
