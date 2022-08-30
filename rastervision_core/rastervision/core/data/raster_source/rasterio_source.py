from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union
import logging
import os
import subprocess
from tempfile import mkdtemp

import numpy as np
import rasterio
from rasterio.enums import (ColorInterp, MaskFlags, Resampling)

from rastervision.pipeline.file_system import download_if_needed
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import (RasterSource, CropOffsets)

if TYPE_CHECKING:
    from rasterio.io import DatasetReader

log = logging.getLogger(__name__)


def build_vrt(vrt_path, image_paths):
    """Build a VRT for a set of TIFF files."""
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    subprocess.run(cmd)


def download_and_build_vrt(image_uris, download_dir):
    log.info('Building VRT...')
    image_paths = [download_if_needed(uri) for uri in image_uris]
    image_path = os.path.join(download_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


def stream_and_build_vrt(images_uris, download_dir):
    log.info('Building VRT...')
    image_paths = images_uris
    image_path = os.path.join(download_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


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
        np.ndarray of shape (height, width, channels) where channels is the
            number of channels in the image_dataset.
    """
    im = image_dataset.read(
        indexes=tuple(bands),
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
                  arr: np.ndarray,
                  fill_value: int = 0) -> np.ndarray:
    """Given a window and corresponding array of values, if the window
    overflows the extent, fill the overflowing regions with fill_value.
    """
    top_overflow = max(0, extent.ymin - window.ymin)
    bottom_overflow = max(0, window.ymax - extent.ymax)
    left_overflow = max(0, extent.xmin - window.xmin)
    right_overflow = max(0, window.xmax - extent.xmax)

    h, w = arr.shape[:2]
    arr[:top_overflow] = fill_value
    arr[h - bottom_overflow:] = fill_value
    arr[:, :left_overflow] = fill_value
    arr[:, w - right_overflow:] = fill_value
    return arr


def get_channel_order_from_dataset(
        image_dataset: 'DatasetReader') -> List[int]:
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
    def __init__(self,
                 uris,
                 raster_transformers=[],
                 download_dir: Optional[str] = None,
                 allow_streaming=False,
                 channel_order=None,
                 extent_crop: Optional[CropOffsets] = None):
        """Constructor.

        This RasterSource can read any file that can be opened by Rasterio/GDAL
        including georeferenced formats such as GeoTIFF and non-georeferenced formats
        such as JPG. See https://www.gdal.org/formats_list.html for more details.

        If channel_order is None, then use non-alpha channels. This also sets any
        masked or NODATA pixel values to be zeros.

        Args:
            channel_order: list of indices of channels to extract from raw imagery
            extent_crop (CropOffsets, optional): Relative
                offsets (top, left, bottom, right) for cropping the extent.
                Useful for using splitting a scene into different datasets.
                Defaults to None i.e. no cropping.
        """
        self.uris = uris
        self.download_dir = mkdtemp() if download_dir is None else download_dir
        self.image_dataset = None
        self.allow_streaming = allow_streaming
        self.extent_crop = extent_crop

        self.imagery_path = self.download_data(
            self.download_dir, stream=self.allow_streaming)
        self.image_dataset = rasterio.open(self.imagery_path)
        self.crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)
        self.dtype = None

        self.height = self.image_dataset.height
        self.width = self.image_dataset.width

        num_channels_raw = self.image_dataset.count
        if channel_order is None:
            channel_order = get_channel_order_from_dataset(self.image_dataset)
        self.bands_to_read = [i + 1 for i in channel_order]

        mask_flags = self.image_dataset.mask_flag_enums
        self.is_masked = any(
            [m for m in mask_flags if m != MaskFlags.all_valid])

        super().__init__(channel_order, num_channels_raw, raster_transformers)

    def download_data(self, download_dir: str, stream: bool = False) -> str:
        """Download any data needed for this Raster Source.

        Return a single local path representing the image or a VRT of the data.
        """
        if len(self.uris) == 1:
            if stream:
                return self.uris[0]
            else:
                return download_if_needed(self.uris[0])
        else:
            if stream:
                return stream_and_build_vrt(self.uris, download_dir)
            else:
                return download_and_build_vrt(self.uris, download_dir)

    def get_crs_transformer(self):
        return self.crs_transformer

    def get_extent(self):
        h, w = self.height, self.width
        if self.extent_crop is not None:
            skip_top, skip_left, skip_bottom, skip_right = self.extent_crop
            ymin, xmin = int(h * skip_top), int(w * skip_left)
            ymax, xmax = h - int(h * skip_bottom), w - int(w * skip_right)
            return Box(ymin, xmin, ymax, xmax)
        return Box(0, 0, h, w)

    def get_dtype(self):
        """Return the numpy.dtype of this scene"""
        if self.dtype is None:
            # Read 1x1 chip to determine dtype
            test_chip = self.get_chip(Box.make_square(0, 0, 1))
            self.dtype = test_chip.dtype
        return self.dtype

    def _get_chip(self,
                  window: Box,
                  out_shape: Optional[Tuple[int, ...]] = None,
                  bands: Optional[Sequence[int]] = None) -> np.ndarray:
        chip = load_window(
            self.image_dataset,
            bands=bands,
            window=window.rasterio_format(),
            is_masked=self.is_masked,
            out_shape=out_shape)
        if self.extent_crop is not None:
            chip = fill_overflow(self.get_extent(), window, chip)
        return chip

    def get_chip(self, window,
                 out_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
        chip = self._get_chip(
            window, out_shape=out_shape, bands=self.bands_to_read)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)
        return chip
