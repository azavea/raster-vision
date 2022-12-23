from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging
import os
import subprocess

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags, Resampling)

from rastervision.pipeline.file_system import download_if_needed, get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.utils import listify_uris

if TYPE_CHECKING:
    from rasterio.io import DatasetReader
    from rastervision.core.data import RasterTransformer

log = logging.getLogger(__name__)


def build_vrt(vrt_path: str, image_uris: List[str]) -> None:
    """Build a VRT for a set of TIFF files.

    Args:
        vrt_path (str): Local path for the VRT to be created.
        image_uris (List[str]): Image URIs.
    """
    log.info('Building VRT...')
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_uris)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris: List[str],
                           vrt_dir: str,
                           stream: bool = False) -> str:
    """Download images (if needed) and build a VRT for a set of TIFF files.

    Args:
        image_uris (List[str]): Image URIs.
        vrt_dir (str): Dir where the VRT will be created.
        stream (bool, optional): If true, do not download images.
            Defaults to False.

    Returns:
        str: The path to the created VRT file.
    """
    if not stream:
        image_uris = [download_if_needed(uri) for uri in image_uris]
    vrt_path = os.path.join(vrt_dir, 'index.vrt')
    build_vrt(vrt_path, image_uris)
    return vrt_path


def load_window(
        image_dataset: 'DatasetReader',
        bands: Optional[Union[int, Sequence[int]]] = None,
        window: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        is_masked: bool = False,
        out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    """Load a window of an image using Rasterio.

    Args:
        image_dataset: a Rasterio dataset.
        bands (Optional[Union[int, Sequence[int]]]): Band index or indices to
            read. Must be 1-indexed.
        window (Optional[Tuple[Tuple[int, int], Tuple[int, int]]]):
            ((row_start, row_stop), (col_start, col_stop)) or
            ((y_min, y_max), (x_min, x_max)). If None, reads the entire raster.
            Defaults to None.
        is_masked (bool): If True, read a masked array from rasterio.
            Defaults to False.
        out_shape (Optional[Tuple[int, int]]): (hieght, width) of the output
            chip. If None, no resizing is done. Defaults to None.

    Returns:
        np.ndarray: array of shape (height, width, channels).
    """
    if bands is not None:
        bands = tuple(bands)
    im = image_dataset.read(
        indexes=bands,
        window=window,
        boundless=True,
        masked=is_masked,
        out_shape=out_shape,
        resampling=Resampling.bilinear)

    if is_masked:
        im = np.ma.filled(im, fill_value=0)

    # Handle non-zero NODATA values by setting the data to 0.
    if bands is None:
        for channel, nodataval in enumerate(image_dataset.nodatavals):
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0
    else:
        for channel, src_band in enumerate(bands):
            src_band_0_indexed = src_band - 1
            nodataval = image_dataset.nodatavals[src_band_0_indexed]
            if nodataval is not None and nodataval != 0:
                im[channel, im[channel] == nodataval] = 0

    im = np.transpose(im, axes=[1, 2, 0])
    return im


def fill_overflow(extent: Box,
                  window: Box,
                  chip: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Where ``chip``'s ``window`` overflows extent, fill with ``fill_value``.

    Args:
        extent (Box): Extent.
        window (Box): Window from which ``chip`` was read.
        chip (np.ndarray): (H, W, C) array.
        fill_value (int, optional): Value to set oveflowing pixels to.
            Defaults to 0.

    Returns:
        np.ndarray: Chip with overflowing regions filled with ``fill_value``.
    """
    top_overflow = max(0, extent.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - extent.ymax)
    left_overflow = max(0, extent.xmin - window.xmin)
    right_overflow = max(0, window.xmax - extent.xmax)

    h, w = chip.shape[:2]
    chip[:top_overflow] = fill_value
    chip[h - bottom_overflow:] = fill_value
    chip[:, :left_overflow] = fill_value
    chip[:, w - right_overflow:] = fill_value
    return chip


def get_channel_order_from_dataset(
        image_dataset: 'DatasetReader') -> List[int]:
    """Get channel order from rasterio image dataset.

    Accounts for dataset's ``colorinterp`` if defined.

    Args:
        image_dataset (DatasetReader): Rasterio image dataset.

    Returns:
        List[int]: List of channel indices.
    """
    colorinterp = image_dataset.colorinterp
    if colorinterp:
        channel_order = [
            i for i, color_interp in enumerate(colorinterp)
            if color_interp != ColorInterp.alpha
        ]
    else:
        channel_order = list(range(0, image_dataset.count))
    return channel_order


class RasterioSource(RasterSource):
    """A rasterio-based :class:`.RasterSource`.

    This RasterSource can read any file that can be opened by
    `Rasterio/GDAL <https://www.gdal.org/formats_list.html>`_.

    If there are multiple image files that cover a single scene, you can pass
    the corresponding list of URIs, and read them as if it were a single
    stitched-together image.

    It can also read non-georeferenced images such as ``.tif``, ``.png``, and
    ``.jpg`` files. This is useful for oblique drone imagery, biomedical
    imagery, and any other (potentially massive!) non-georeferenced images.

    If channel_order is None, then use non-alpha channels. This also sets any
    masked or NODATA pixel values to be zeros.
    """

    def __init__(self,
                 uris: Union[str, List[str]],
                 raster_transformers: List['RasterTransformer'] = [],
                 allow_streaming: bool = False,
                 channel_order: Optional[Sequence[int]] = None,
                 extent: Optional[Box] = None,
                 tmp_dir: Optional[str] = None):
        """Constructor.

        Args:
            uris (Union[str, List[str]]): One or more URIs of images. If more
                than one, the images will be mosaiced together using GDAL.
            raster_transformers (List['RasterTransformer']): RasterTransformers
                to use to trasnform chips after they are read.
            allow_streaming (bool): If True, read data without downloading the
                entire file first. Defaults to False.
            channel_order (Optional[Sequence[int]]): List of indices of
                channels to extract from raw imagery. Can be a subset of the
                available channels. If None, all channels available in the
                image will be read. Defaults to None.
            extent (Optional[Box], optional): Use-specified extent. If None,
                the full extent of the raster source is used.
            tmp_dir (Optional[str]): Directory to use for storing the VRT
                (needed if multiple uris or allow_streaming=True). If None,
                will be auto-generated. Defaults to None.
        """
        self.uris = listify_uris(uris)
        self.allow_streaming = allow_streaming
        self._num_channels = None
        self._dtype = None

        self.tmp_dir = tmp_dir
        if self.tmp_dir is None:
            self._tmp_dir = get_tmp_dir()
            self.tmp_dir = self._tmp_dir.name

        self.imagery_path = self.download_data(
            self.tmp_dir, stream=self.allow_streaming)
        self.image_dataset = rasterio.open(self.imagery_path)

        block_shapes = set(self.image_dataset.block_shapes)
        if len(block_shapes) > 1:
            log.warn('Raster bands have non-identical block shapes: '
                     f'{block_shapes}. This can slow down reading. '
                     'Consider re-tiling using GDAL.')

        for h, w in block_shapes:
            # the choice of 4 here is arbitrary
            if max(h, w) / min(h, w) > 4:
                log.warn(f'Raster block size {(h, w)} is too non-square. '
                         'This can slow down reading. '
                         'Consider re-tiling using GDAL.')

        self._crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)

        num_channels_raw = self.image_dataset.count
        if channel_order is None:
            channel_order = get_channel_order_from_dataset(self.image_dataset)
        self.bands_to_read = np.array(channel_order, dtype=int) + 1

        # number of output channels
        if len(raster_transformers) == 0:
            self._num_channels = len(self.bands_to_read)

        mask_flags = self.image_dataset.mask_flag_enums
        self.is_masked = any(m for m in mask_flags if m != MaskFlags.all_valid)

        if extent is None:
            height = self.image_dataset.height
            width = self.image_dataset.width
            extent = Box(0, 0, height, width)

        super().__init__(
            channel_order,
            num_channels_raw,
            raster_transformers=raster_transformers,
            extent=extent)

    @property
    def num_channels(self) -> int:
        """Number of channels in the chips read from this source.

        .. note::

            Unlike the parent class, ``RasterioSource`` applies channel_order
            (via ``bands_to_read``) before ``raster_transformers``. So the
            number of output channels is not guaranteed to be equal to
            ``len(channel_order)``.
        """
        if self._num_channels is None:
            self._set_info_from_chip()
        return self._num_channels

    @property
    def dtype(self) -> Tuple[int, int, int]:
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

    def download_data(self,
                      vrt_dir: Optional[str] = None,
                      stream: bool = False) -> str:
        """Download any data needed for this raster source.

        Return a single local path representing the image or a VRT of the data.
        """
        if len(self.uris) == 1:
            if stream:
                return self.uris[0]
            else:
                return download_if_needed(self.uris[0])
        else:
            if vrt_dir is None:
                raise ValueError('vrt_dir is required if using >1 image URIs.')
            return download_and_build_vrt(self.uris, vrt_dir, stream=stream)

    def _get_chip(self,
                  window: Box,
                  bands: Optional[Sequence[int]] = None,
                  out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        window = window.shift_origin(self.extent)
        chip = load_window(
            self.image_dataset,
            bands=bands,
            window=window.rasterio_format(),
            is_masked=self.is_masked,
            out_shape=out_shape)
        chip = fill_overflow(self.extent, window, chip)
        return chip

    def get_chip(self,
                 window: Box,
                 bands: Optional[Union[Sequence[int], slice]] = None,
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
            the output chip. If None, no resizing is done. Defaults to None.

        Returns:
            np.ndarray: A chip of shape (height, width, channels).
        """
        bands_to_read = self.bands_to_read
        if bands is not None:
            bands_to_read = bands_to_read[bands]
        chip = self._get_chip(window, out_shape=out_shape, bands=bands_to_read)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)
        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)
        elif isinstance(key, slice):
            key = [key]
        elif isinstance(key, tuple):
            pass
        else:
            raise TypeError('Unsupported key type.')

        slices = list(key)
        assert 1 <= len(slices) <= 3
        assert all(s is not None for s in slices)
        assert isinstance(slices[0], slice)
        if len(slices) == 1:
            h, = slices
            w = slice(None, None)
            c = None
        elif len(slices) == 2:
            assert isinstance(slices[1], slice)
            h, w = slices
            c = None
        else:
            h, w, c = slices

        if any(x is not None and x < 0
               for x in [h.start, h.stop, w.start, w.stop]):
            raise NotImplementedError()

        ymin, xmin, ymax, xmax = self.extent
        _ymin = 0 if h.start is None else h.start
        _xmin = 0 if w.start is None else w.start
        _ymax = ymax if h.stop is None else h.stop
        _xmax = xmax if w.stop is None else w.stop
        window = Box(_ymin, _xmin, _ymax, _xmax)

        out_shape = None
        if h.step is not None or w.step is not None:
            out_h, out_w = window.size
            if h.step is not None:
                out_h //= h.step
            if w.step is not None:
                out_w //= w.step
            out_shape = (int(out_h), int(out_w))

        chip = self.get_chip(window, bands=c, out_shape=out_shape)
        return chip
