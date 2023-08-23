from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging

import numpy as np
from xarray import DataArray

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.utils import parse_array_slices_Nd, fill_overflow

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer, CRSTransformer

log = logging.getLogger(__name__)


class XarraySource(RasterSource):
    """A RasterSource for reading an Xarry DataArray.

    .. warning:: ``XarraySource`` API is in beta.
    """

    def __init__(self,
                 data_array: DataArray,
                 crs_transformer: 'CRSTransformer',
                 raster_transformers: List['RasterTransformer'] = [],
                 channel_order: Optional[Sequence[int]] = None,
                 num_channels_raw: Optional[int] = None,
                 bbox: Optional[Box] = None,
                 temporal: bool = False):
        """Constructor.

        Args:
            uris (Union[str, List[str]]): One or more URIs of images. If more
                than one, the images will be mosaiced together using GDAL.
            crs_transformer (CRSTransformer): A CRSTransformer defining the
                mapping between pixel and map coords.
            raster_transformers (List['RasterTransformer']): RasterTransformers
                to use to transform chips after they are read.
            channel_order (Optional[Sequence[int]]): List of indices of
                channels to extract from raw imagery. Can be a subset of the
                available channels. If None, all channels available in the
                image will be read. Defaults to None.
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If None, the full extent available in the source file is used.
            temporal (bool): If True, data_array is expected to have a "time"
                dimension and the chips returned will be of shape (T, H, W, C).
        """
        self.temporal = temporal
        if self.temporal:
            if set(data_array.dims) != {'x', 'y', 'band', 'time'}:
                raise ValueError(
                    'If temporal=True, data_array must have 4 dimensions: '
                    '"x", "y", "band", and "time" (in any order).')
        else:
            if set(data_array.dims) != {'x', 'y', 'band'}:
                raise ValueError(
                    'If temporal=False, data_array must have 3 dimensions: '
                    '"x", "y", and "band" (in any order).')

        self.data_array = data_array.transpose(..., 'y', 'x', 'band')
        self.ndim = data_array.ndim
        self._crs_transformer = crs_transformer

        if num_channels_raw is None:
            num_channels_raw = len(data_array.band)
        if channel_order is None:
            channel_order = np.arange(num_channels_raw, dtype=int)
        else:
            channel_order = np.array(channel_order, dtype=int)
        self._num_channels = None
        self._dtype = None
        if len(raster_transformers) == 0:
            self._num_channels = len(channel_order)
            self._dtype = data_array.dtype

        height, width = len(data_array.y), len(data_array.x)
        self.full_extent = Box(0, 0, height, width)
        if bbox is None:
            bbox = self.full_extent

        super().__init__(
            channel_order,
            num_channels_raw,
            raster_transformers=raster_transformers,
            bbox=bbox)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of the raster as a (height, width, num_channels) tuple."""
        H, W = self.bbox.size
        if self.temporal:
            T = len(self.data_array.time)
            return T, H, W, self.num_channels
        return H, W, self.num_channels

    @property
    def num_channels(self) -> int:
        """Number of channels in the chips read from this source.

        .. note::

            Unlike the parent class, ``XarraySource`` applies
            ``channel_order`` before ``raster_transformers``. So the number of
            output channels is not guaranteed to be equal to
            ``len(channel_order)``.
        """
        if self._num_channels is None:
            self._set_info_from_chip()
        return self._num_channels

    @property
    def dtype(self) -> np.dtype:
        if self._dtype is None:
            self._set_info_from_chip()
        return self._dtype

    @property
    def crs_transformer(self) -> RasterioCRSTransformer:
        return self._crs_transformer

    def _set_info_from_chip(self):
        """Read 1x1 chip to get info not statically inferrable."""
        test_chip = self.get_chip(Box(0, 0, 1, 1))
        self._dtype = test_chip.dtype
        self._num_channels = test_chip.shape[-1]

    def _get_chip(self,
                  window: Box,
                  bands: Union[int, Sequence[int], slice] = slice(None),
                  time: Union[int, Sequence[int], slice] = slice(None),
                  out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        window = window.to_global_coords(self.bbox)

        yslice, xsclice = window.to_slices()
        if self.temporal:
            chip = self.data_array.isel(
                x=xsclice, y=yslice, band=bands, time=time).to_numpy()
        else:
            chip = self.data_array.isel(
                x=xsclice, y=yslice, band=bands).to_numpy()

        *batch_dims, h, w, c = chip.shape
        if window.size != (h, w):
            window_actual = window.intersection(self.full_extent)
            yslice, xsclice = window_actual.to_local_coords(window).to_slices()
            tmp = np.zeros((*batch_dims, *window.size, c))
            tmp[..., yslice, xsclice, :] = chip
            chip = tmp

        chip = fill_overflow(self.bbox, window, chip)
        if out_shape is not None:
            chip = self.resize(chip, out_shape)
        return chip

    def get_chip(self,
                 window: Box,
                 bands: Optional[Union[int, Sequence[int], slice]] = None,
                 time: Union[int, Sequence[int], slice] = slice(None),
                 out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        """Read a chip specified by a window from the file.

        Args:
            window (Box): Bounding box of chip in pixel coordinates.
            bands (Optional[Union[Sequence[int], slice]], optional): Subset of
                bands to read. Note that this will be applied on top of the
                channel_order (if specified). So if this is an RGB image and
                channel_order=[2, 1, 0], then using bands=[0] will return the
                B-channel. Defaults to None.
            out_shape (Optional[Tuple[int, ...]], optional): (hieght, width) of
                the output chip. If None, no resizing is done.
                Defaults to None.

        Returns:
            np.ndarray: A chip of shape (height, width, channels).
        """
        if bands is None or bands == slice(None):
            bands = self.channel_order
        else:
            bands = self.channel_order[bands]
        chip = self._get_chip(
            window, bands=bands, time=time, out_shape=out_shape)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, bands)
        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, dim_slices = parse_array_slices_Nd(
            key, extent=self.extent, dims=self.ndim)
        if self.temporal:
            t, h, w, c = dim_slices
        else:
            h, w, c = dim_slices
            t = None

        out_shape = None
        if h.step is not None or w.step is not None:
            out_h, out_w = window.size
            if h.step is not None:
                out_h //= h.step
            if w.step is not None:
                out_w //= w.step
            out_shape = (int(out_h), int(out_w))

        chip = self.get_chip(window, bands=c, time=t, out_shape=out_shape)
        return chip
