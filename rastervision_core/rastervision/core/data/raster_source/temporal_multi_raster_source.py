from typing import TYPE_CHECKING, Any, Sequence

from pydantic import NonNegativeInt as NonNegInt
import numpy as np

from rastervision.core.box import Box
from rastervision.core.data.raster_source import (RasterSource,
                                                  MultiRasterSource)
from rastervision.core.data.utils import all_equal, parse_array_slices_Nd

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer


class TemporalMultiRasterSource(MultiRasterSource):
    """Merge multiple ``RasterSources`` by stacking them along a new dim."""

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 primary_source_idx: NonNegInt = 0,
                 raster_transformers: Sequence['RasterTransformer'] = [],
                 bbox: Box | None = None):
        """Constructor.

        Args:
            raster_sources: Sequence of RasterSources.
            primary_source_idx: Index of the raster source whose CRS, dtype,
                and other attributes will override those of the other raster
                sources.
            raster_transformers: Sequence of transformers. Defaults to ``[]``.
            bbox: User-specified crop of the extent. If given, the primary
                raster source's bbox is set to this. If ``None``, the full
                extent available in the source file of the primary raster
                source is used.
        """
        if not all_equal([rs.num_channels for rs in raster_sources]):
            raise ValueError(
                'All sub raster sources must have the same num_channels.')

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        primary_rs = raster_sources[primary_source_idx]
        dtype_raw = primary_rs.dtype
        num_channels_raw = primary_rs.num_channels_raw
        channel_order = None

        if bbox is None:
            bbox = primary_rs.bbox
        else:
            primary_rs.set_bbox(bbox)

        RasterSource.__init__(
            self,
            channel_order=channel_order,
            num_channels_raw=num_channels_raw,
            dtype_raw=dtype_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

        self.raster_sources = raster_sources
        self.primary_source_idx = primary_source_idx

        self.non_primary_sources = [
            rs for rs in self.raster_sources if rs != self.primary_source
        ]

        self.validate_raster_sources()

    @classmethod
    def from_stac(cls, *args, **kwargs):
        """Not implemented for ``TemporalMultiRasterSource``."""
        raise NotImplementedError(
            'Create raster sources by calling MultiRasterSource.from_stac() '
            'on each Item and then pass them to TemporalMultiRasterSource.')

    def _get_chip(self, window: Box,
                  out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Get chip w/o applying channel_order and transformers.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            np.ndarray: 4D array of shape (T, H, W, C).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.stack(sub_chips)
        return chip

    def get_chip(self, window: Box,
                 out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Return the transformed chip in the window.

        Get processed chips from sub raster sources (with their respective
        channel orders and transformations applied), stack them along a new
        temporal dimension, apply channel_order, followed by transformations.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            np.ndarray: 4D array of shape (T, H, W, C).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.stack(sub_chips)

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip)

        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, (t, h, w, c) = parse_array_slices_Nd(
            key, extent=self.extent, dims=4)
        chip = self.get_chip(window)
        if h.step is not None or w.step is not None:
            chip = chip[:, ::h.step, ::w.step, :]
        chip = chip[t, ...]
        chip = chip[..., c]

        return chip

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (len(self.raster_sources), *self.primary_source.shape)
