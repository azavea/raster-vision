from typing import TYPE_CHECKING, Any
from abc import ABC, abstractmethod

import numpy as np
from skimage.transform import resize

from rastervision.core.box import Box
from rastervision.core.data.utils import parse_array_slices_Nd
from rastervision.core.data.raster_transformer.utils import (
    get_transformed_num_channels, get_transformed_dtype)

if TYPE_CHECKING:
    from rastervision.core.data import (CRSTransformer, RasterTransformer)


class ChannelOrderError(Exception):
    def __init__(self, channel_order: list[int], num_channels_raw: int):
        self.channel_order = channel_order
        self.num_channels_raw = num_channels_raw
        msg = (f'The channel_order ({channel_order}) contains an '
               f'index >= num_channels_raw ({num_channels_raw}).')
        super().__init__(msg)


class RasterSource(ABC):
    """A source of raster data.

    This should be subclassed when adding a new source of raster data such as
    a set of files, an API, a TMS URI schema, etc.
    """

    def __init__(self,
                 *,
                 channel_order: list[int] | None,
                 num_channels_raw: int,
                 dtype_raw: np.dtype,
                 bbox: Box,
                 raster_transformers: list['RasterTransformer'] = []):
        """Constructor.

        Args:
            channel_order: list of channel indices to use when extracting chip
                from raw imagery.
            num_channels_raw: Number of channels in the raw imagery before
                applying channel_order.
            dtype_raw: dtype of the raw imagery before applying transformers.
            bbox: Extent or a crop of the extent.
            raster_transformers: ``RasterTransformers`` for transforming chips
                whenever they are retrieved. Defaults to ``[]``.
        """
        if channel_order is None:
            channel_order = list(range(num_channels_raw))

        if any(c >= num_channels_raw for c in channel_order):
            raise ChannelOrderError(channel_order, num_channels_raw)

        self.channel_order = channel_order
        self.num_channels_raw = num_channels_raw
        self.dtype_raw = np.dtype(dtype_raw)
        self.raster_transformers = raster_transformers
        self._bbox = bbox
        self._num_channels = None
        self._dtype = None

    @property
    def num_channels(self) -> int:
        """Number of channels in the chips read from this source."""
        if self._num_channels is None:
            num_channels = len(self.channel_order)
            self._num_channels = get_transformed_num_channels(
                self.raster_transformers, num_channels)
        return self._num_channels

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the raster as a (height, width, num_channels) tuple."""
        H, W = self.bbox.size
        return H, W, self.num_channels

    @property
    def dtype(self) -> 'np.dtype':
        """``numpy.dtype`` of the chips read from this source."""
        if self._dtype is None:
            dtype = self.dtype_raw
            self._dtype = get_transformed_dtype(self.raster_transformers,
                                                dtype)
        return self._dtype

    @property
    def bbox(self) -> 'Box':
        """Bounding box applied to the source imagery."""
        return self._bbox

    @property
    def extent(self) -> 'Box':
        """Extent of the ``RasterSource``."""
        return self.bbox.extent

    @property
    @abstractmethod
    def crs_transformer(self) -> 'CRSTransformer':
        """Associated :class:`.CRSTransformer`."""

    def set_bbox(self, bbox: 'Box') -> None:
        """Set self.bbox to the given value.

        .. note:: This method is idempotent.

        Args:
            bbox (Box): User-specified bbox in pixel coordinates.
        """
        self._bbox = bbox

    @abstractmethod
    def _get_chip(self,
                  window: 'Box',
                  out_shape: tuple[int, int] | None = None) -> 'np.ndarray':
        """Return raw chip without applying channel_order or transforms.

        Args:
            window (Box): The window for which to get the chip.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            [height, width, channels] numpy array
        """

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, (h, w, c) = parse_array_slices_Nd(
            key, extent=self.extent, dims=3)
        chip = self.get_chip(window)
        if h.step is not None or w.step is not None:
            chip = chip[::h.step, ::w.step]
        chip = chip[..., c]

        return chip

    def get_chip(self, window: 'Box',
                 out_shape: tuple[int, int] | None = None) -> 'np.ndarray':
        """Return the transformed chip in the window.

        Get a raw chip, extract subset of channels using channel_order, and then apply
        transformations.

        Args:
            window (Box): The window for which to get the chip.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            np.ndarray: Array of shape (..., height, width, channels).
        """
        chip = self._get_chip(window, out_shape=out_shape)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip)

        return chip

    def get_chip_by_map_window(self, window_map_coords: 'Box', *args,
                               **kwargs) -> 'np.ndarray':
        """Same as get_chip(), but input is a window in map coords."""
        window_pixel_coords = self.crs_transformer.map_to_pixel(
            window_map_coords, bbox=self.bbox).normalize()
        chip = self.get_chip(window_pixel_coords, *args, **kwargs)
        return chip

    def _get_chip_by_map_window(self, window_map_coords: 'Box', *args,
                                **kwargs) -> 'np.ndarray':
        """Same as _get_chip(), but input is a window in map coords."""
        window_pixel_coords = self.crs_transformer.map_to_pixel(
            window_map_coords, bbox=self.bbox)
        chip = self._get_chip(window_pixel_coords, *args, **kwargs)
        return chip

    def get_raw_chip(self,
                     window: 'Box',
                     out_shape: tuple[int, int] | None = None) -> 'np.ndarray':
        """Return raw chip without applying channel_order or transforms.

        Args:
            window (Box): The window for which to get the chip.

        Returns:
            np.ndarray: Array of shape (..., height, width, channels).
        """
        return self._get_chip(window, out_shape=out_shape)

    def resize(self,
               chip: 'np.ndarray',
               out_shape: tuple[int, int] | None = None) -> 'np.ndarray':
        out_shape = chip.shape[:-3] + out_shape
        out = resize(chip, out_shape, preserve_range=True, anti_aliasing=True)
        out = out.round(6).astype(chip.dtype)
        return out
