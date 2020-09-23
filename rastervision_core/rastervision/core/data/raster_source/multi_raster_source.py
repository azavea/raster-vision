from typing import Optional, Sequence, Union
from pydantic import conint

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import ActivateMixin
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.crs_transformer import CRSTransformer
from rastervision.core.data.utils import all_equal


class MultiRasterSourceError(Exception):
    pass


class MultiRasterSource(ActivateMixin, RasterSource):
    """A RasterSource that combines multiple RasterSources by concatenting
    their output along the channel dimension (assumed to be the last dimension).
    """

    def __init__(
            self,
            raster_sources: Sequence[RasterSource],
            raw_channel_order: Sequence[conint(ge=0)],
            force_subchip_size_fill_value: Optional[Union[int, float]] = None,
            force_same_dtype: bool = False,
            channel_order: Optional[Sequence[conint(ge=0)]] = None,
            crs_source: conint(ge=0) = 0,
            raster_transformers: Sequence = []):
        """Constructor.

        Args:
            raster_sources (Sequence[RasterSource]): Sequence of RasterSources.
            raw_channel_order (Sequence[conint(ge=0)]): Channel ordering that
                will always be applied before channel_order.
            force_subchip_size_fill_value (Optional[Union[int, float]]):
                Value used to pad sub-chips so that they are all the same size as the
                sub-chip from the first sub-source.  This is to accommodate small
                differences in sub-raster size such those that might be caused by
                attempting to reproject disparate sub-rasters into the same projection.
                No special reprojection logic is triggered by this parameter.  Use with
                caution.
            force_same_dtype (bool): If true, force all subchips to have the same dtype
                as the first subchip.  No careful converstion is done, just a quick cast.
                Use with caution.
            channel_order (Sequence[conint(ge=0)], optional): Channel ordering
                that will be used by .get_chip(). Defaults to None.
            raster_transformers (Sequence, optional): Sequence of transformers.
                Defaults to [].
        """
        num_channels = len(raw_channel_order)
        if not channel_order:
            channel_order = list(range(num_channels))

        super().__init__(channel_order, num_channels, raster_transformers)

        self.force_subchip_size_fill_value = force_subchip_size_fill_value
        self.force_same_dtype = force_same_dtype
        self.raster_sources = raster_sources
        self.raw_channel_order = list(raw_channel_order)
        self.crs_source = crs_source

        self.validate_raster_sources()

    def validate_raster_sources(self) -> None:
        dtypes = [rs.get_dtype() for rs in self.raster_sources]
        if not self.force_same_dtype and not all_equal(dtypes):
            raise MultiRasterSourceError(
                'dtypes of all sub raster sources must be equal. '
                f'Got: {dtypes} '
                '(carfully consider using force_same_dtype)')

        extents = [rs.get_extent() for rs in self.raster_sources]
        if self.force_subchip_size_fill_value is None and not all_equal(
                extents):
            raise MultiRasterSourceError(
                'extents of all sub raster sources must be equal. '
                f'Got: {extents} '
                '(carefully consider using force_subchip_size_fill_value)')

        sub_num_channels = sum(rs.num_channels for rs in self.raster_sources)
        if sub_num_channels != self.num_channels:
            raise MultiRasterSourceError(
                f'num_channels ({self.num_channels}) != sum of num_channels '
                f'of sub raster sources ({sub_num_channels})')

    def _subcomponents_to_activate(self) -> None:
        return self.raster_sources

    def get_extent(self) -> Box:
        rs = self.raster_sources[0]
        extent = rs.get_extent()
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
        chip_slices = [rs._get_chip(window) for rs in self.raster_sources]

        if self.force_same_dtype is not True:
            for i in range(1, len(chip_slices)):
                chip_slices[i] = chip_slices[i].astype(chip_slices[0].dtype)

        if self.force_subchip_size_fill_value is not None:
            (w1, h1, ch1) = chip_slices[0].shape
            for i in range(1, len(chip_slices)):
                (w2, h2, ch2) = chip_slices[i].shape
                if w1 != w2 or h1 != h2:
                    a = np.ndarray((w1, h1, ch2), dtype=chip_slices[i].dtype)
                    a.fill(self.force_subchip_size_fill_value)
                    a[0:min(w1, w2) - 1, 0:min(h1, h2) - 1, :] = chip_slices[
                        i][0:min(w1, w2) - 1, 0:min(h1, h2) - 1, :]
                    chip_slices[i] = a

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
        chip_slices = [rs.get_chip(window) for rs in self.raster_sources]

        if self.force_same_dtype:
            for i in range(1, len(chip_slices)):
                chip_slices[i] = chip_slices[i].astype(chip_slices[0].dtype)

        if self.force_subchip_size_fill_value is not None:
            (w1, h1, ch1) = chip_slices[0].shape
            for i in range(1, len(chip_slices)):
                (w2, h2, ch2) = chip_slices[i].shape
                if w1 != w2 or h1 != h2:
                    a = np.ndarray((w1, h1, ch2), dtype=chip_slices[i].dtype)
                    a.fill(self.force_subchip_size_fill_value)
                    a[0:min(w1, w2) - 1, 0:min(h1, h2) - 1, :] = chip_slices[
                        i][0:min(w1, w2) - 1, 0:min(h1, h2) - 1, :]
                    chip_slices[i] = a

        chip = np.concatenate(chip_slices, axis=-1)
        chip = chip[..., self.raw_channel_order]
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip
