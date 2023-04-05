from typing import TYPE_CHECKING, Any, List, Optional, Tuple
from abc import ABC, abstractmethod, abstractproperty

from rastervision.core.box import Box
from rastervision.core.data.utils import parse_array_slices

if TYPE_CHECKING:
    from rastervision.core.data import (CRSTransformer, RasterTransformer)
    import numpy as np


class ChannelOrderError(Exception):
    def __init__(self, channel_order: List[int], num_channels_raw: int):
        self.channel_order = channel_order
        self.num_channels_raw = num_channels_raw
        msg = (f'The channel_order ({channel_order}) contains an'
               f'index >= num_channels_raw ({num_channels_raw}).')
        super().__init__(msg)


class RasterSource(ABC):
    """A source of raster data.

    This should be subclassed when adding a new source of raster data such as
    a set of files, an API, a TMS URI schema, etc.
    """

    def __init__(self,
                 channel_order: Optional[List[int]],
                 num_channels_raw: int,
                 raster_transformers: List['RasterTransformer'] = [],
                 extent: Optional[Box] = None):
        """Constructor.

        Args:
            channel_order: list of channel indices to use when extracting chip
                from raw imagery.
            num_channels_raw: Number of channels in the raw imagery before
                applying channel_order.
            raster_transformers: ``RasterTransformers`` for transforming chips
                whenever they are retrieved. Defaults to ``[]``.
            extent: User-specified extent. If None, the full extent of the
                raster source is used.
        """
        if channel_order is None:
            channel_order = list(range(num_channels_raw))

        if any(c >= num_channels_raw for c in channel_order):
            raise ChannelOrderError(channel_order, num_channels_raw)

        self.channel_order = channel_order
        self.num_channels_raw = num_channels_raw
        self.raster_transformers = raster_transformers
        self._extent = extent

    @property
    def num_channels(self) -> int:
        """Number of channels in the chips read from this source."""
        return len(self.channel_order)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the raster as a (height, width, num_channels) tuple."""
        ymin, xmin, ymax, xmax = self.extent
        return ymax - ymin, xmax - xmin, self.num_channels

    @abstractproperty
    def dtype(self) -> 'np.dtype':
        """``numpy.dtype`` of the chips read from this source."""
        pass

    @property
    def extent(self) -> 'Box':
        """Extent of the RasterSource."""
        return self._extent

    @abstractproperty
    def crs_transformer(self) -> 'CRSTransformer':
        """Associated :class:`.CRSTransformer`."""
        pass

    def set_extent(self, extent: 'Box') -> None:
        """Set self.extent to the given value.

        .. note:: This method is idempotent.

        Args:
            extent (Box): User-specified extent in pixel coordinates.
        """
        self._extent = extent

    @abstractmethod
    def _get_chip(self, window: 'Box') -> 'np.ndarray':
        """Return raw chip without applying channel_order or transforms.

        Args:
            window: Box

        Returns:
            [height, width, channels] numpy array
        """
        pass

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, (h, w, c) = parse_array_slices(key, extent=self.extent, dims=3)
        chip = self.get_chip(window)
        if h.step is not None or w.step is not None:
            chip = chip[::h.step, ::w.step]
        chip = chip[..., c]

        return chip

    def get_chip(self, window: 'Box') -> 'np.ndarray':
        """Return the transformed chip in the window.

        Get a raw chip, extract subset of channels using channel_order, and then apply
        transformations.

        Args:
            window (Box): The window for which to get the chip.

        Returns:
            np.ndarray: Array of shape (height, width, channels).
        """
        chip = self._get_chip(window)

        chip = chip[:, :, self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)

        return chip

    def get_raw_chip(self, window: 'Box') -> 'np.ndarray':
        """Return raw chip without applying channel_order or transforms.

        Args:
            window (Box): The window for which to get the chip.

        Returns:
            np.ndarray: Array of shape (height, width, channels).
        """
        return self._get_chip(window)

    def get_image_array(self) -> 'np.ndarray':
        """Return entire transformed image array.

        .. warning:: Not safe to call on very large RasterSources.

        Returns:
            np.ndarray: Array of shape (height, width, channels).
        """
        return self.get_chip(self.extent)

    def get_raw_image_array(self) -> 'np.ndarray':
        """Return raw image for the full extent.

        .. warning:: Not safe to call on very large RasterSources.

        Returns:
            np.ndarray: Array of shape (height, width, channels).
        """
        return self.get_raw_chip(self.extent)
