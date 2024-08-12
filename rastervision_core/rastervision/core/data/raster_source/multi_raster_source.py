from typing import TYPE_CHECKING, Sequence

from pydantic import NonNegativeInt as NonNegInt
import numpy as np
from pystac import Item

from rastervision.core.box import Box
from rastervision.core.data.raster_source import RasterSource, RasterioSource
from rastervision.core.data.raster_source.stac_config import subset_assets
from rastervision.core.data.utils import all_equal

if TYPE_CHECKING:
    from typing import Self
    from rastervision.core.data import RasterTransformer, CRSTransformer


class MultiRasterSource(RasterSource):
    """
    Merge multiple ``RasterSources`` by concatenating along the channel dim.
    """

    def __init__(self,
                 raster_sources: Sequence[RasterSource],
                 primary_source_idx: NonNegInt = 0,
                 channel_order: Sequence[NonNegInt] | None = None,
                 raster_transformers: Sequence['RasterTransformer'] = [],
                 bbox: Box | None = None):
        """Constructor.

        Args:
            raster_sources: Sequence of RasterSources.
            primary_source_idx: Index of the raster source whose CRS, dtype,
                and other attributes will override those of the other raster
                sources.
            channel_order: Channel ordering that will be used by
                :meth:`.MultiRasterSource.get_chip`. Defaults to ``None``.
            raster_transformers: List of transformers. Defaults to ``[]``.
            bbox: User-specified crop of the extent. If specified, the primary
                raster source's bbox is set to this. If ``None``, the full
                extent available in the source file of the primary raster
                source is used.
        """
        dtype_raw = raster_sources[primary_source_idx].dtype

        num_channels_raw = sum(rs.num_channels for rs in raster_sources)
        if not channel_order:
            num_channels = sum(rs.num_channels for rs in raster_sources)
            channel_order = list(range(num_channels))

        # validate primary_source_idx
        if not (0 <= primary_source_idx < len(raster_sources)):
            raise IndexError('primary_source_idx must be in range '
                             '[0, len(raster_sources)].')

        if bbox is None:
            bbox = raster_sources[primary_source_idx].bbox
        else:
            raster_sources[primary_source_idx].set_bbox(bbox)

        super().__init__(
            channel_order=channel_order,
            num_channels_raw=num_channels_raw,
            dtype_raw=dtype_raw,
            bbox=bbox,
            raster_transformers=raster_transformers)

        self.raster_sources = raster_sources
        self.primary_source_idx = primary_source_idx
        self.non_primary_sources = [
            rs for i, rs in enumerate(raster_sources)
            if i != primary_source_idx
        ]
        self.validate_raster_sources()

    @classmethod
    def from_stac(
            cls,
            item: Item,
            assets: list[str] | None,
            primary_source_idx: NonNegInt = 0,
            raster_transformers: list['RasterTransformer'] = [],
            channel_order: Sequence[int] | None = None,
            bbox: Box | tuple[int, int, int, int] | None = None,
            bbox_map_coords: Box | tuple[int, int, int, int] | None = None,
            allow_streaming: bool = False) -> 'Self':
        """Construct a ``MultiRasterSource`` from a STAC Item.

        This creates a :class:`.RasterioSource` for each asset and puts all
        the raster sources together into a ``MultiRasterSource``. If ``assets``
        is not specified, all the assets in the STAC item are used.

        Only assets that are readable by rasterio are supported.

        Args:
            item: STAC Item.
            assets: List of names of assets to use. If ``None``, all assets
                present in the item will be used. Defaults to ``None``.
            primary_source_idx: Index of the raster source whose CRS, dtype,
                and other attributes will override those of the other raster
                sources.
            raster_transformers: RasterTransformers to use to transform chips
                after they are read. Defaults to ``[]``.
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
            allow_streaming: Passed to :class:`.RasterioSource`. If ``False``,
                assets will be downloaded. Defaults to ``True``.
        """
        if bbox is not None and bbox_map_coords is not None:
            raise ValueError('Specify either bbox or bbox_map_coords, '
                             'but not both.')

        if assets is not None:
            item = subset_assets(item, assets)

        uris = [asset.href for asset in item.assets.values()]
        raster_sources = [
            RasterioSource(uri, allow_streaming=allow_streaming)
            for uri in uris
        ]

        crs_transformer = raster_sources[primary_source_idx].crs_transformer
        if bbox_map_coords is not None:
            bbox_map_coords = Box(*bbox_map_coords)
            bbox = crs_transformer.map_to_pixel(bbox_map_coords).normalize()
        elif bbox is not None:
            bbox = Box(*bbox)

        raster_source = MultiRasterSource(
            raster_sources,
            primary_source_idx=primary_source_idx,
            raster_transformers=raster_transformers,
            channel_order=channel_order,
            bbox=bbox)
        return raster_source

    def validate_raster_sources(self) -> None:
        """Validate sub-``RasterSources``.

        Checks if all raster sources have the same dtype.
        """
        dtypes = [rs.dtype for rs in self.raster_sources]
        if not all_equal(dtypes):
            raise ValueError(
                'dtypes of all sub raster sources must be the same. '
                f'Got: {dtypes}.')

    @property
    def primary_source(self) -> RasterSource:
        """Primary sub-``RasterSource``"""
        return self.raster_sources[self.primary_source_idx]

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the raster as a (..., H, W, C) tuple."""
        *shape, _ = self.primary_source.shape
        return (*shape, self.num_channels)

    @property
    def crs_transformer(self) -> 'CRSTransformer':
        return self.primary_source.crs_transformer

    def _get_sub_chips(self,
                       window: Box,
                       out_shape: tuple[int, int] | None = None
                       ) -> list[np.ndarray]:
        """Return chips from sub raster sources as a list.

        If all extents are identical, simply retrieves chips from each sub
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
            window: The window for which to get the chip, in pixel coordinates.
            out_shape: (height, width) to resize the chip to.

        Returns:
            List of chips from each sub raster source.
        """

        def get_chip(rs: RasterSource,
                     window: Box,
                     map: bool = False,
                     out_shape: tuple[int, int] | None = None) -> np.ndarray:
            if map:
                func = rs.get_chip_by_map_window
            else:
                func = rs.get_chip
            return func(window, out_shape=out_shape)

        primary_rs = self.primary_source
        other_rses = self.non_primary_sources

        primary_sub_chip = get_chip(primary_rs, window, out_shape=out_shape)
        if out_shape is None:
            out_shape = primary_sub_chip.shape[:2]
        window_map_coords = primary_rs.crs_transformer.pixel_to_map(
            window, bbox=primary_rs.bbox)
        sub_chips = [
            get_chip(rs, window_map_coords, map=True, out_shape=out_shape)
            for rs in other_rses
        ]
        sub_chips.insert(self.primary_source_idx, primary_sub_chip)

        return sub_chips

    def _get_chip(self, window: Box,
                  out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Get chip w/o applying channel_order and transformers.

        Args:
            window (Box): The window for which to get the chip, in pixel
                coordinates.
            out_shape (tuple[int, int] | None): (height, width) to resize
                the chip to.

        Returns:
            Array of shape (height, width, channels).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        return chip

    def get_chip(self, window: Box,
                 out_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Return the transformed chip in the window.

        Get processed chips from sub raster sources (with their respective
        channel orders and transformations applied), concatenate them along the
        channel dimension, apply channel_order, followed by transformations.

        Args:
            window: The window for which to get the chip, in pixel coordinates.
            out_shape: (height, width) to resize the chip to.

        Returns:
            Array of shape (height, width, channels).
        """
        sub_chips = self._get_sub_chips(window, out_shape=out_shape)
        chip = np.concatenate(sub_chips, axis=-1)
        chip = chip[..., self.channel_order]

        for transformer in self.raster_transformers:
            chip = transformer.transform(chip)

        return chip
