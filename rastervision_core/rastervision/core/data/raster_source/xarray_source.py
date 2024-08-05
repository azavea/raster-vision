from typing import TYPE_CHECKING, Any, Sequence
import logging

import numpy as np
from xarray import DataArray

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.utils import parse_array_slices_Nd, fill_overflow

if TYPE_CHECKING:
    from pystac import Item, ItemCollection
    from rastervision.core.data import RasterTransformer, CRSTransformer

log = logging.getLogger(__name__)


class XarraySource(RasterSource):
    """A RasterSource for reading an Xarry DataArray."""

    def __init__(self,
                 data_array: DataArray,
                 crs_transformer: 'CRSTransformer',
                 raster_transformers: list['RasterTransformer'] = [],
                 channel_order: Sequence[int] | None = None,
                 bbox: Box | None = None,
                 temporal: bool = False):
        """Constructor.

        Args:
            uris: One or more URIs of images. If more than one, the images will
                be mosaiced together using GDAL.
            crs_transformer: A CRSTransformer defining the mapping between
                pixel and map coords.
            raster_transformers: RasterTransformers to use to transform chips
                after they are read.
            channel_order: List of indices of channels to extract from raw
                imagery. Can be a subset of the available channels. If
                ``None``, all channels available in the image will be read.
                Defaults to ``None``.
            bbox: User-specified crop of the extent. If ``None``, the full
                extent available in the source file is used.
            temporal: If ``True``, data_array is expected to have a "time"
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
        dtype_raw = data_array.dtype

        num_channels_raw = len(data_array.band)
        if channel_order is None:
            channel_order = np.arange(num_channels_raw, dtype=int)
        else:
            channel_order = np.array(channel_order, dtype=int)

        height, width = len(data_array.y), len(data_array.x)
        self.full_extent = Box(0, 0, height, width)
        if bbox is None:
            bbox = self.full_extent
        else:
            if bbox not in self.full_extent:
                new_bbox = bbox.intersection(self.full_extent)
                log.warning(f'Clipping ({bbox}) to the DataArray\'s '
                            f'full extent ({self.full_extent}). '
                            f'New bbox={new_bbox}')
                bbox = new_bbox

        super().__init__(
            channel_order=channel_order,
            num_channels_raw=num_channels_raw,
            dtype_raw=dtype_raw,
            raster_transformers=raster_transformers,
            bbox=bbox)

    @classmethod
    def from_stac(
            cls,
            item_or_item_collection: 'Item | ItemCollection',
            raster_transformers: list['RasterTransformer'] = [],
            channel_order: Sequence[int] | None = None,
            bbox: Box | tuple[int, int, int, int] | None = None,
            bbox_map_coords: Box | tuple[int, int, int, int] | None = None,
            temporal: bool = False,
            allow_streaming: bool = False,
            stackstac_args: dict = dict(rescale=False)) -> 'XarraySource':
        """Construct an ``XarraySource`` from a STAC Item or ItemCollection.

        Args:
            item_or_item_collection: STAC Item or ItemCollection.
            raster_transformers: RasterTransformers to use to transform chips
                after they are read.
            channel_order: List of indices of channels to extract from raw
                imagery. Can be a subset of the available channels. If None,
                all channels available in the image will be read.
                Defaults to None.
            bbox: User-specified crop of the extent. Can be :class:`.Box` or
                (ymin, xmin, ymax, xmax) tuple. If None, the full extent
                available in the source file is used. Mutually exclusive with
                ``bbox_map_coords``. Defaults to ``None``.
            bbox_map_coords: User-specified bbox in EPSG:4326 coords. Can be
                :class:`.Box` or (ymin, xmin, ymax, xmax) tuple. Useful for
                cropping the raster source so that only part of the raster is
                read from. Mutually exclusive with ``bbox``.
                Defaults to ``None``.
            temporal: If True, data_array is expected to have a "time"
                dimension and the chips returned will be of shape (T, H, W, C).
            allow_streaming: If False, load the entire DataArray into memory.
                Defaults to True.
            stackstac_args: Optional arguments to pass to stackstac.stack().
        """
        import stackstac

        data_array = stackstac.stack(item_or_item_collection, **stackstac_args)

        if not temporal and 'time' in data_array.dims:
            if len(data_array.time) > 1:
                raise ValueError('temporal=False but len(data_array.time) > 1')
            data_array = data_array.isel(time=0)

        if not allow_streaming:
            from humanize import naturalsize
            log.info('Loading the full DataArray into memory '
                     f'({naturalsize(data_array.nbytes)}).')
            data_array.load()

        crs_transformer = RasterioCRSTransformer(
            transform=data_array.transform, image_crs=data_array.crs)

        if bbox is not None:
            if bbox_map_coords is not None:
                raise ValueError('Specify either bbox or bbox_map_coords, '
                                 'but not both.')
            bbox = Box(*bbox)
        elif bbox_map_coords is not None:
            bbox_map_coords = Box(*bbox_map_coords)
            bbox = crs_transformer.map_to_pixel(bbox_map_coords).normalize()
        else:
            bbox = None

        raster_source = XarraySource(
            data_array,
            crs_transformer=crs_transformer,
            raster_transformers=raster_transformers,
            channel_order=channel_order,
            bbox=bbox,
            temporal=temporal)
        return raster_source

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the raster as a (height, width, num_channels) tuple."""
        H, W = self.bbox.size
        if self.temporal:
            T = len(self.data_array.time)
            return T, H, W, self.num_channels
        return H, W, self.num_channels

    @property
    def crs_transformer(self) -> RasterioCRSTransformer:
        return self._crs_transformer

    def _get_chip(self,
                  window: Box,
                  bands: int | Sequence[int] | slice = slice(None),
                  time: int | Sequence[int] | slice = slice(None),
                  out_shape: tuple[int, ...] | None = None) -> np.ndarray:
        window = window.to_global_coords(self.bbox)

        window_within_bbox = window.intersection(self.bbox)

        yslice, xslice = window_within_bbox.to_slices()
        if self.temporal:
            chip = self.data_array.isel(
                x=xslice, y=yslice, band=bands, time=time).to_numpy()
        else:
            chip = self.data_array.isel(
                x=xslice, y=yslice, band=bands).to_numpy()

        if window != window_within_bbox:
            *batch_dims, h, w, c = chip.shape
            # coords of window_within_bbox within window
            yslice, xslice = window_within_bbox.to_local_coords(
                window).to_slices()
            tmp = np.zeros((*batch_dims, *window.size, c))
            tmp[..., yslice, xslice, :] = chip
            chip = tmp

        chip = fill_overflow(self.bbox, window, chip)
        if out_shape is not None:
            chip = self.resize(chip, out_shape)
        return chip

    def get_chip(self,
                 window: Box,
                 bands: int | Sequence[int] | slice | None = None,
                 time: int | Sequence[int] | slice = slice(None),
                 out_shape: tuple[int, ...] | None = None) -> np.ndarray:
        """Read a chip specified by a window from the file.

        Args:
            window: Bounding box of chip in pixel coordinates.
            bands: Subset of bands to read. Note that this will be applied on
                top of the ``channel_order`` (if specified). So if this is an
                RGB image and ``channel_order=[2, 1, 0]``, then using
                ``bands=[0]`` will return the B-channel. Defaults to ``None``.
            out_shape: (height, width) of the output chip. If ``None``, no
                resizing is done. Defaults to ``None``.

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
            chip = transformer.transform(chip)
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
