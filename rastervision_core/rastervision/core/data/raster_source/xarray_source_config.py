from typing import Any, Dict, Optional, Tuple, Union
import logging

from rastervision.pipeline.config import Field, register_config
from rastervision.core.data.raster_source.raster_source_config import (
    RasterSourceConfig)
from rastervision.core.data.raster_source.stac_config import (
    STACItemConfig, STACItemCollectionConfig)
from rastervision.core.data.raster_source.xarray_source import XarraySource

log = logging.getLogger(__name__)


@register_config('xarray_source')
class XarraySourceConfig(RasterSourceConfig):
    """Configure an :class:`.XarraySource`."""

    stac: Union[STACItemConfig, STACItemCollectionConfig] = Field(
        ...,
        description='STAC Item or ItemCollection to build the DataArray from.')
    allow_streaming: bool = Field(
        True,
        description='If False, load the entire DataArray into memory. '
        'Defaults to True.')
    bbox_map_coords: Optional[Tuple[float, float, float, float]] = Field(
        None,
        description='Optional user-specified bbox in EPSG:4326 coords of the '
        'form (ymin, xmin, ymax, xmax). Useful for cropping the raster source '
        'so that only part of the raster is read from. This is ignored if '
        'bbox is also specified. Defaults to None.')
    temporal: bool = Field(
        False, description='Whether the data is a time-series.')
    stackstac_args: Dict[str, Any] = Field(
        {}, description='Optional arguments to pass to stackstac.stack().')

    def build(self,
              tmp_dir: Optional[str] = None,
              use_transformers: bool = True) -> XarraySource:
        item_or_item_collection = self.stac.build()
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])
        raster_source = XarraySource.from_stac(
            item_or_item_collection,
            raster_transformers=raster_transformers,
            channel_order=self.channel_order,
            bbox=self.bbox,
            bbox_map_coords=self.bbox_map_coords,
            temporal=self.temporal,
            allow_streaming=self.allow_streaming,
            stackstac_args=self.stackstac_args,
        )
        return raster_source
