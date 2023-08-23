from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import logging

import numpy as np
import rasterio

from rastervision.pipeline.file_system import download_if_needed, get_tmp_dir
from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.raster_source import RasterSource
from rastervision.core.data.utils import (listify_uris, parse_array_slices_Nd)
from rastervision.core.data.utils.raster import fill_overflow
from rastervision.core.data.utils.rasterio import (
    read_window, get_channel_order_from_dataset, download_and_build_vrt,
    is_masked)

if TYPE_CHECKING:
    from rastervision.core.data import RasterTransformer

log = logging.getLogger(__name__)


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
                 bbox: Optional[Box] = None,
                 tmp_dir: Optional[str] = None):
        """Constructor.

        Args:
            uris (Union[str, List[str]]): One or more URIs of images. If more
                than one, the images will be mosaiced together using GDAL.
            raster_transformers (List['RasterTransformer']): RasterTransformers
                to use to transform chips after they are read.
            allow_streaming (bool): If True, read data without downloading the
                entire file first. Defaults to False.
            channel_order (Optional[Sequence[int]]): List of indices of
                channels to extract from raw imagery. Can be a subset of the
                available channels. If None, all channels available in the
                image will be read. Defaults to None.
            bbox (Optional[Box], optional): User-specified crop of the extent.
                If None, the full extent available in the source file is used.
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

        self._crs_transformer = RasterioCRSTransformer.from_dataset(
            self.image_dataset)

        num_channels_raw = self.image_dataset.count
        if channel_order is None:
            channel_order = get_channel_order_from_dataset(self.image_dataset)
        self.bands_to_read = np.array(channel_order, dtype=int) + 1
        self.is_masked = is_masked(self.image_dataset)

        # number of output channels
        if len(raster_transformers) == 0:
            self._num_channels = len(self.bands_to_read)

        height = self.image_dataset.height
        width = self.image_dataset.width
        if bbox is None:
            bbox = Box(0, 0, height, width)

        super().__init__(
            channel_order,
            num_channels_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

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
        window = window.to_global_coords(self.bbox)
        chip = read_window(
            self.image_dataset,
            bands=bands,
            window=window.rasterio_format(),
            is_masked=self.is_masked,
            out_shape=out_shape)
        chip = fill_overflow(self.bbox, window, chip)
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
                the output chip. If None, no resizing is done.
                Defaults to None.

        Returns:
            np.ndarray: A chip of shape (height, width, channels).
        """
        if bands is None or bands == slice(None):
            bands = self.bands_to_read
        else:
            bands = self.bands_to_read[bands]
        chip = self._get_chip(window, out_shape=out_shape, bands=bands)
        for transformer in self.raster_transformers:
            chip = transformer.transform(chip, self.channel_order)
        return chip

    def __getitem__(self, key: Any) -> 'np.ndarray':
        if isinstance(key, Box):
            return self.get_chip(key)

        window, (h, w, c) = parse_array_slices_Nd(
            key, extent=self.extent, dims=3)

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
