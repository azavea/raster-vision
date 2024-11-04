from typing import TYPE_CHECKING, Any, Callable
import logging

from pyproj import Transformer
from pyproj.exceptions import ProjError
import numpy as np
import rasterio as rio
from rasterio.transform import (rowcol, xy)
from rasterio.windows import bounds, from_bounds, Window
from rasterio import Affine

from rastervision.core.box import Box
from rastervision.core.data.crs_transformer import (CRSTransformer,
                                                    IdentityCRSTransformer)

if TYPE_CHECKING:
    from typing import Self

log = logging.getLogger(__name__)

PIXEL_PRECISION = 6
MAP_PRECISION = 6


def pyproj_wrapper(
        func: Callable[..., tuple[Any, Any]],
        from_crs: str,
        to_crs: str,
) -> Callable[..., tuple[Any, Any]]:
    # For some transformations, pyproj attempts to download transformation
    # grids from the internet for improved accuracy when
    # Transformer.transform() is called. If it fails to connect to the
    # internet, it silently returns (inf, inf) and silently modifies its
    # behavior to not access the internet on subsequent calls, causing
    # them to succeed (though possibly with a loss of accuracy). See
    # https://github.com/pyproj4/pyproj/issues/705 for details.
    #
    # The below workaround forces an error to be raised by setting
    # errcheck=True and ignoring the first error.
    def _wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs, errcheck=True)
        except ProjError as e:
            log.debug(f'pyproj: {e}')
            if 'network' in str(e).lower():
                log.warning(
                    'pyproj tried and failed to connect to the internet to '
                    'download transformation grids for the transformation from'
                    f'\n{from_crs}\nto\n{to_crs}.\nSee '
                    'https://github.com/pyproj4/pyproj/issues/705 for details.'
                )
            return func(*args, **kwargs, errcheck=True)

    return _wrapper


class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource."""

    def __init__(self,
                 transform: Affine,
                 image_crs: str,
                 map_crs: str = 'epsg:4326',
                 round_pixels: bool = True):
        """Constructor.

        Args:
            transform: Rasterio affine transform.
            image_crs: CRS of image in format that PyProj can handle eg. WKT or
                init string.
            map_crs: CRS of the labels. Defaults to "epsg:4326".
            round_pixels: If ``True``, round outputs of :meth:`.map_to_pixel`.
                Defaults to ``False``.
        """

        if (image_crs is None) or (image_crs == map_crs):
            self.map2image = lambda *args, **kws: args[:2]
            self.image2map = lambda *args, **kws: args[:2]
        else:
            self._map2image = Transformer.from_crs(
                map_crs, image_crs, always_xy=True).transform
            self._image2map = Transformer.from_crs(
                image_crs, map_crs, always_xy=True).transform
            self.map2image = pyproj_wrapper(self._map2image, map_crs,
                                            image_crs)
            self.image2map = pyproj_wrapper(self._image2map, image_crs,
                                            map_crs)

        self.round_pixels = round_pixels

        super().__init__(transform, image_crs, map_crs)

    def __repr__(self) -> str:
        cls_name = type(self).__name__

        image_crs_str = str(self.image_crs)
        if len(image_crs_str) > 70:
            image_crs_str = image_crs_str[:70] + '...'

        map_crs_str = str(self.map_crs)
        if len(map_crs_str) > 70:
            map_crs_str = map_crs_str[:70] + '...'

        transform_str = (
            '\n\t\t' + (str(self.transform).replace('\n', '\n\t\t')))
        out = f"""{cls_name}(
            image_crs="{image_crs_str}",
            map_crs="{map_crs_str}",
            round_pixels={self.round_pixels},
            transform={transform_str})
        """
        return out

    def _map_to_pixel_point(
            self,
            map_point: tuple[float, float] | tuple[np.ndarray, np.ndarray]
    ) -> tuple[int, int] | tuple[np.ndarray, np.ndarray]:
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = self.map2image(*map_point)
        x, y = image_point
        row, col = rowcol(
            self.transform, x, y, op=lambda x: np.round(x, PIXEL_PRECISION))
        if self.round_pixels:
            row, col = np.round(row), np.round(col)
        pixel_point = (col, row)
        return pixel_point

    def _map_to_pixel_box(self, box: Box) -> Box:
        ymin, xmin, ymax, xmax = box
        xmin, ymin = self.map2image(xmin, ymin)
        xmax, ymax = self.map2image(xmax, ymax)
        rio_window = from_bounds(
            left=xmin,
            bottom=ymin,
            right=xmax,
            top=ymax,
            transform=self.transform,
        )
        (ymin, ymax), (xmin, xmax) = rio_window.toranges()
        ymin, xmin, ymax, xmax = (
            round(ymin, PIXEL_PRECISION),
            round(xmin, PIXEL_PRECISION),
            round(ymax, PIXEL_PRECISION),
            round(xmax, PIXEL_PRECISION),
        )
        if self.round_pixels:
            ymin, xmin, ymax, xmax = (
                round(ymin),
                round(xmin),
                round(ymax),
                round(xmax),
            )
        pixel_box = Box(ymin, xmin, ymax, xmax)
        return pixel_box

    def _pixel_to_map_point(
            self, pixel_point: tuple[int, int] | tuple[np.ndarray, np.ndarray]
    ) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in map coordinates
        """
        col, row = pixel_point
        x, y = xy(self.transform, row, col, offset='center')
        map_point = self.image2map(x, y)
        return map_point

    def _pixel_to_map_box(self, box: Box) -> Box:
        rio_window = Window(*box.to_xywh())
        xmin, ymin, xmax, ymax = bounds(rio_window, transform=self.transform)
        xmin, ymin, xmax, ymax = (
            round(xmin, PIXEL_PRECISION),
            round(ymin, PIXEL_PRECISION),
            round(xmax, PIXEL_PRECISION),
            round(ymax, PIXEL_PRECISION),
        )
        xmin, ymin = self.image2map(xmin, ymin)
        xmax, ymax = self.image2map(xmax, ymax)
        map_box = Box(ymin, xmin, ymax, xmax)
        return map_box

    @classmethod
    def from_dataset(cls,
                     dataset: Any,
                     map_crs: str | None = 'epsg:4326',
                     **kwargs) -> 'IdentityCRSTransformer | Self':
        """Build from rasterio dataset.

        Args:
            dataset: Rasterio dataset.
            map_crs: Target map CRS. Defaults to 'epsg:4326'.
            **kwargs: Extra args for :meth:`.__init__`.
        """
        transform = dataset.transform
        image_crs = None if dataset.crs is None else dataset.crs.wkt
        map_crs = image_crs if map_crs is None else map_crs

        no_crs_tf = (image_crs is None) or (image_crs == map_crs)
        no_affine_tf = (transform is None) or (transform == Affine.identity())
        if no_crs_tf and no_affine_tf:
            return IdentityCRSTransformer()

        if transform is None:
            transform = Affine.identity()

        return cls(transform, image_crs, map_crs, **kwargs)

    @classmethod
    def from_uri(cls, uri: str, map_crs: str | None = 'epsg:4326',
                 **kwargs) -> 'IdentityCRSTransformer | Self':
        """Build from raster URI.

        Args:
            uri: Raster URI.
            map_crs: Target map CRS. Defaults to 'epsg:4326'.
            **kwargs: Extra args for :meth:`.__init__`.
        """
        with rio.open(uri) as ds:
            return cls.from_dataset(ds, map_crs=map_crs, **kwargs)
