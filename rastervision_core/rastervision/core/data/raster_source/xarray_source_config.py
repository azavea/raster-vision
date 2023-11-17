from typing import Optional, Tuple, Union
import logging

from rastervision.pipeline.config import Field, register_config
from rastervision.core.box import Box
from rastervision.core.data.raster_source.raster_source_config import (
    RasterSourceConfig)
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source.stac_config import (
    STACItemConfig, STACItemCollectionConfig)
from rastervision.core.data.raster_source.xarray_source import (XarraySource)

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

    def build(self,
              tmp_dir: Optional[str] = None,
              use_transformers: bool = True) -> XarraySource:
        import stackstac

        item_or_item_collection = self.stac.build()
        data_array = stackstac.stack(item_or_item_collection)

        if not self.temporal and 'time' in data_array.dims:
            if len(data_array.time) > 1:
                raise ValueError('temporal=False but len(data_array.time) > 1')
            data_array = data_array.isel(time=0)

        if not self.allow_streaming:
            from humanize import naturalsize
            log.info('Loading the full DataArray into memory '
                     f'({naturalsize(data_array.nbytes)}).')
            data_array.load()

        crs_transformer = RasterioCRSTransformer(
            transform=data_array.transform, image_crs=data_array.crs)
        raster_transformers = ([rt.build() for rt in self.transformers]
                               if use_transformers else [])

        if self.bbox is not None:
            if self.bbox_map_coords is not None:
                log.info('Using bbox and ignoring bbox_map_coords.')
            bbox = Box(*self.bbox)
        elif self.bbox_map_coords is not None:
            bbox_map_coords = Box(*self.bbox_map_coords)
            bbox = crs_transformer.map_to_pixel(bbox_map_coords).normalize()
        else:
            bbox = None

        raster_source = XarraySource(
            data_array,
            crs_transformer=crs_transformer,
            raster_transformers=raster_transformers,
            channel_order=self.channel_order,
            bbox=bbox,
            temporal=self.temporal)
        return raster_source
