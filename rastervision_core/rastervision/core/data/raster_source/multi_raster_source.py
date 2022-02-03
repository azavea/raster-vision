from typing import Optional, Sequence
from pydantic import conint

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import ActivateMixin
from rastervision.core.data.raster_source import (RasterSource, CropOffsets)
from rastervision.core.data.crs_transformer import CRSTransformer
from rastervision.core.data.raster_source.rasterio_source import RasterioSource
from rastervision.core.data.utils import all_equal


class MultiRasterSourceError(Exception):
    pass


class MultiRasterSource(ActivateMixin, RasterSource):
    """A RasterSource that combines multiple RasterSources by concatenting
    their output along the channel dimension (assumed to be the last dimension).
    """

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 raw_channel_order: Sequence[conint(ge=0)],
                 force_same_dtype: bool = False,
                 channel_order: Optional[Sequence[conint(ge=0)]] = None,
                 crs_source: conint(ge=0) = 0,
                 raster_transformers: Sequence = [],
                 extent_crop: Optional[CropOffsets] = None):
        """Constructor.

        Args:
            raster_sources (Sequence[RasterSource]): Sequence of RasterSources.
            raw_channel_order (Sequence[conint(ge=0)]): Channel ordering that
                will always be applied before channel_order.
            force_same_dtype (bool): If true, force all subchips to have the same dtype
                as the first subchip.  No careful conversion is done, just a quick cast.
                Use with caution.
            channel_order (Sequence[conint(ge=0)], optional): Channel ordering
                that will be used by .get_chip(). Defaults to None.
            raster_transformers (Sequence, optional): Sequence of transformers.
                Defaults to [].
            extent_crop (CropOffsets, optional): Relative
                offsets (top, left, bottom, right) for cropping the extent.
                Useful for using splitting a scene into different datasets.
                Defaults to None i.e. no cropping.
        """
        num_channels = len(raw_channel_order)
        if not channel_order:
            channel_order = list(range(num_channels))

        super().__init__(channel_order, num_channels, raster_transformers)

        self.force_same_dtype = force_same_dtype
        self.raster_sources = raster_sources
        self.raw_channel_order = list(raw_channel_order)
        self.crs_source = crs_source
        self.extent_crop = extent_crop

        self.extents = [rs.get_extent() for rs in self.raster_sources]
        self.all_extents_equal = all_equal(self.extents)

        self.validate_raster_sources()

    def validate_raster_sources(self) -> None:
        dtypes = [rs.get_dtype() for rs in self.raster_sources]
        if not self.force_same_dtype and not all_equal(dtypes):
            raise MultiRasterSourceError(
                'dtypes of all sub raster sources must be equal. '
                f'Got: {dtypes} '
                '(carfully consider using force_same_dtype)')
        if not self.all_extents_equal:
            all_rasterio_sources = all(
                isinstance(rs, RasterioSource) for rs in self.raster_sources)
            if not all_rasterio_sources:
                raise MultiRasterSourceError(
                    'Non-identical extents are only supported '
                    'for RasterioSource raster sources.')

        sub_num_channels = sum(
            len(rs.channel_order) for rs in self.raster_sources)
        if sub_num_channels != self.num_channels:
            raise MultiRasterSourceError(
                f'num_channels ({self.num_channels}) != sum of num_channels '
                f'of sub raster sources ({sub_num_channels})')

    def _subcomponents_to_activate(self) -> None:
        return self.raster_sources

    def get_extent(self) -> Box:
        rs = self.raster_sources[0]
        extent = rs.get_extent()
        if self.extent_crop is not None:
            h, w = extent.get_height(), extent.get_width()
            skip_top, skip_left, skip_bottom, skip_right = self.extent_crop
            ymin, xmin = int(h * skip_top), int(w * skip_left)
            ymax, xmax = h - int(h * skip_bottom), w - int(w * skip_right)
            return Box(ymin, xmin, ymax, xmax)
        return extent

    def get_dtype(self) -> np.dtype:
        rs = self.raster_sources[0]
        dtype = rs.get_dtype()
        return dtype

    def get_crs_transformer(self) -> CRSTransformer:
        rs = self.raster_sources[self.crs_source]
        return rs.get_crs_transformer()

    def _get_chip(self, window: Box) -> np.ndarray:
        """Return the raw chip located in the window.

        Get raw chips from sub raster sources, concatenate them and
        apply raw_channel_order.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        if self.all_extents_equal:
            chip_slices = [rs._get_chip(window) for rs in self.raster_sources]
        else:
            primary_rs = self.raster_sources[0]
            chip_slice = primary_rs._get_chip(window)
            out_shape = chip_slice.shape
            world_window = primary_rs.get_transformed_window(window)
            pixel_windows = [
                rs.get_transformed_window(world_window, inverse=True)
                for rs in self.raster_sources[1:]
            ]
            chip_slices = [chip_slice] + [
                rs._get_chip(w, out_shape=out_shape)
                for rs, w in zip(self.raster_sources[1:], pixel_windows)
            ]

        if self.force_same_dtype:
            dtype = chip_slices[0].dtype
            for i in range(1, len(chip_slices)):
                chip_slices[i] = chip_slices[i].astype(dtype)

        chip = np.concatenate(chip_slices, axis=-1)
        chip = chip[..., self.raw_channel_order]
        return chip

    def get_chip(self, window: Box) -> np.ndarray:
        """Return the transformed chip in the window.

        Get raw chips from sub raster sources, concatenate them,
        apply raw_channel_order, followed by channel_order, followed
        by transformations.

        Args:
            window: Box

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        if self.all_extents_equal:
            chip_slices = [rs.get_chip(window) for rs in self.raster_sources]
        else:
            primary_rs = self.raster_sources[0]
            chip_slice = primary_rs.get_chip(window)
            out_shape = chip_slice.shape[:2]
            world_window = primary_rs.get_transformed_window(window)
            pixel_windows = [
                rs.get_transformed_window(world_window, inverse=True)
                for rs in self.raster_sources[1:]
            ]
            chip_slices = [chip_slice] + [
                rs.get_chip(w, out_shape=out_shape)
                for rs, w in zip(self.raster_sources[1:], pixel_windows)
            ]

        if self.force_same_dtype:
            dtype = chip_slices[0].dtype
            for i in range(1, len(chip_slices)):
                chip_slices[i] = chip_slices[i].astype(dtype)

        chip = np.concatenate(chip_slices, axis=-1)
        chip = chip[..., self.raw_channel_order]
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip
