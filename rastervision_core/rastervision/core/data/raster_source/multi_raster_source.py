from typing import Optional, Sequence, List, Tuple
from pydantic import conint

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.crs_transformer import CRSTransformer
from rastervision.core.data.raster_source.rasterio_source import RasterioSource
from rastervision.core.data.utils import all_equal


class MultiRasterSource(RasterSource):
    """Merge multiple ``RasterSources`` by concatenating along channel dim."""

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 primary_source_idx: conint(ge=0) = 0,
                 force_same_dtype: bool = False,
                 channel_order: Optional[Sequence[conint(ge=0)]] = None,
                 raster_transformers: Sequence = [],
                 extent: Optional[Box] = None):
        """Constructor.

        Args:
            raster_sources (Sequence[RasterSource]): Sequence of RasterSources.
            primary_source_idx (0 <= int < len(raster_sources)): Index of the
                raster source whose CRS, dtype, and other attributes will
                override those of the other raster sources.
            force_same_dtype (bool): If true, force all sub-chips to have the
                same dtype as the primary_source_idx-th sub-chip. No careful
                conversion is done, just a quick cast. Use with caution.
            channel_order (Sequence[conint(ge=0)], optional): Channel ordering
                that will be used by .get_chip(). Defaults to None.
            raster_transformers (Sequence, optional): Sequence of transformers.
                Defaults to [].
        """
        num_channels_raw = sum(rs.num_channels_raw for rs in raster_sources)
        if not channel_order:
            num_channels = sum(rs.num_channels for rs in raster_sources)
            channel_order = list(range(num_channels))

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        if extent is None:
            extent = raster_sources[primary_source_idx].extent

        super().__init__(
            channel_order,
            num_channels_raw,
            raster_transformers=raster_transformers,
            extent=extent)

        self.force_same_dtype = force_same_dtype
        self.raster_sources = raster_sources
        self.primary_source_idx = primary_source_idx

        self.extents = [rs.extent for rs in self.raster_sources]
        self.all_extents_equal = all_equal(self.extents)

        self.validate_raster_sources()

    def validate_raster_sources(self) -> None:
        """Validate sub-``RasterSources``.

        Checks if:

        - dtypes are same or ``force_same_dtype`` is True.
        - each sub-``RasterSource`` is a :class:`.RasterioSource` if extents
          not identical.

        """
        dtypes = [rs.dtype for rs in self.raster_sources]
        if not self.force_same_dtype and not all_equal(dtypes):
            raise ValueError(
                'dtypes of all sub raster sources must be the same. '
                f'Got: {dtypes} '
                '(Use force_same_dtype to cast all to the dtype of the '
                'primary source)')
        if not self.all_extents_equal:
            all_rasterio_sources = all(
                isinstance(rs, RasterioSource) for rs in self.raster_sources)
            if not all_rasterio_sources:
                raise NotImplementedError(
                    'Non-identical extents are only '
                    'supported for RasterioSource raster sources.')

    @property
    def primary_source(self) -> RasterSource:
        """Primary sub-``RasterSource``"""
        return self.raster_sources[self.primary_source_idx]

    @property
    def dtype(self) -> np.dtype:
        return self.primary_source.dtype

    @property
    def crs_transformer(self) -> CRSTransformer:
        return self.primary_source.crs_transformer

    def _get_sub_chips(self, window: Box,
                       raw: bool = False) -> List[np.ndarray]:
        """If all extents are identical, simply retrieves chips from each sub
        raster source. Otherwise, follows the following algorithm
            - using pixel-coords window, get chip from the primary sub raster
            source
            - convert window to world coords using the CRS of the primary sub
            raster source
            - for each remaining sub raster source
                - convert world-coords window to pixel coords using the sub
                raster source's CRS
                - get chip from the sub raster source using this window;
                specify `out_shape` when reading to ensure shape matches
                reference chip from the primary sub raster source

        Args:
            window (Box): window to read, in pixel coordinates.
            raw (bool, optional): If True, uses RasterSource._get_chip.
                Otherwise, RasterSource.get_chip. Defaults to False.

        Returns:
            List[np.ndarray]: List of chips from each sub raster source.
        """

        def get_chip(
                rs: RasterSource,
                window: Box,
                out_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
            if raw:
                return rs._get_chip(window, out_shape=out_shape)
            return rs.get_chip(window, out_shape=out_shape)

        if self.all_extents_equal:
            sub_chips = [get_chip(rs, window) for rs in self.raster_sources]
        else:
            primary_rs = self.primary_source
            other_rses = [rs for rs in self.raster_sources if rs != primary_rs]

            primary_sub_chip = get_chip(primary_rs, window)
            out_shape = primary_sub_chip.shape[:2]
            world_window = primary_rs.crs_transformer.pixel_to_map(window)
            pixel_windows = [
                rs.crs_transformer.map_to_pixel(world_window)
                for rs in other_rses
            ]
            sub_chips = [
                get_chip(rs, w, out_shape=out_shape)
                for rs, w in zip(other_rses, pixel_windows)
            ]
            sub_chips.insert(self.primary_source_idx, primary_sub_chip)

        if self.force_same_dtype:
            dtype = sub_chips[self.primary_source_idx].dtype
            sub_chips = [chip.astype(dtype) for chip in sub_chips]

        return sub_chips

    def _get_chip(self, window: Box) -> np.ndarray:
        """Return the raw chip located in the window.

        Get raw chips from sub raster sources and concatenate them.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        sub_chips = self._get_sub_chips(window, raw=True)
        chip = np.concatenate(sub_chips, axis=-1)
        return chip

    def get_chip(self, window: Box) -> np.ndarray:
        """Return the transformed chip in the window.

        Get processed chips from sub raster sources (with their respective
        channel orders and transformations applied), concatenate them along the
        channel dimension, apply channel_order, followed by transformations.

        Args:
            window: Box

        Returns:
            np.ndarray with shape [height, width, channels]
        """
        sub_chips = self._get_sub_chips(window, raw=False)
        chip = np.concatenate(sub_chips, axis=-1)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip
