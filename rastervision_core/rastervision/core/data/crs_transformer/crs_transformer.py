from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, overload, Tuple

from rasterio.windows import Window
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry

from rastervision.core.box import Box

if TYPE_CHECKING:
    import numpy as np


class CRSTransformer(ABC):
    """Transforms map points in some CRS into pixel coordinates.

    Each transformer is associated with a particular :class:`.RasterSource`.
    """

    def __init__(self,
                 transform: Optional[Any] = None,
                 image_crs: Optional[str] = None,
                 map_crs: Optional[str] = None):
        self.transform = transform
        self.image_crs = image_crs
        self.map_crs = map_crs

    @overload
    def map_to_pixel(self, inp: Tuple[float, float]) -> Tuple[int, int]:
        ...

    @overload
    def map_to_pixel(self, inp: Tuple['np.array', 'np.array']
                     ) -> Tuple['np.array', 'np.array']:
        ...

    @overload
    def map_to_pixel(self, inp: Box) -> Box:
        ...

    @overload
    def map_to_pixel(self, inp: BaseGeometry) -> BaseGeometry:
        ...

    def map_to_pixel(self, inp):
        """Transform input from pixel to map coords.

        Args:
            inp: (x, y) tuple or Box or rasterio Window or shapely geometry in
                pixel coordinates. If tuple, x and y can be single values or
                array-like.

        Returns:
            Coordinate-transformed input in the same format.
        """
        if isinstance(inp, Box):
            box_in = inp
            ymin, xmin, ymax, xmax = box_in
            xmin_tf, ymin_tf = self._map_to_pixel((xmin, ymin))
            xmax_tf, ymax_tf = self._map_to_pixel((xmax, ymax))
            box_out = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
            return box_out
        elif isinstance(inp, Window):
            window_in = inp
            (ymin, ymax), (xmin, xmax) = window_in.toranges()
            xmin_tf, ymin_tf = self._map_to_pixel((xmin, ymin))
            xmax_tf, ymax_tf = self._map_to_pixel((xmax, ymax))
            window_out = Window.from_slices(
                slice(ymin_tf, ymax_tf), slice(xmin_tf, xmax_tf))
            return window_out
        elif isinstance(inp, BaseGeometry):
            geom_in = inp
            geom_out = transform(
                lambda x, y, z=None: self._map_to_pixel((x, y)), geom_in)
            return geom_out
        elif len(inp) == 2:
            return self._map_to_pixel(inp)
        else:
            raise TypeError('Input must be 2-tuple or Box or rasterio Window '
                            'or shapely geometry.')

    @overload
    def pixel_to_map(self, inp: Tuple[float, float]) -> Tuple[float, float]:
        ...

    @overload
    def pixel_to_map(self, inp: Tuple['np.array', 'np.array']
                     ) -> Tuple['np.array', 'np.array']:
        ...

    @overload
    def pixel_to_map(self, inp: Box) -> Box:
        ...

    @overload
    def pixel_to_map(self, inp: BaseGeometry) -> BaseGeometry:
        ...

    def pixel_to_map(self, inp):
        """Transform input from pixel to map coords.

        Args:
            inp: (x, y) tuple or Box or rasterio Window or shapely geometry in
                pixel coordinates. If tuple, x and y can be single values or
                array-like.

        Returns:
            Coordinate-transformed input in the same format.
        """
        if isinstance(inp, Box):
            box_in = inp
            ymin, xmin, ymax, xmax = box_in
            xmin_tf, ymin_tf = self._pixel_to_map((xmin, ymin))
            xmax_tf, ymax_tf = self._pixel_to_map((xmax, ymax))
            box_out = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
            return box_out
        elif isinstance(inp, Window):
            window_in = inp
            (ymin, ymax), (xmin, xmax) = window_in.toranges()
            xmin_tf, ymin_tf = self._pixel_to_map((xmin, ymin))
            xmax_tf, ymax_tf = self._pixel_to_map((xmax, ymax))
            window_out = Window.from_slices(
                slice(ymin_tf, ymax_tf), slice(xmin_tf, xmax_tf))
            return window_out
        elif isinstance(inp, BaseGeometry):
            geom_in = inp
            geom_out = transform(
                lambda x, y, z=None: self._pixel_to_map((x, y)), geom_in)
            return geom_out
        elif len(inp) == 2:
            return self._pixel_to_map(inp)
        else:
            raise TypeError('Input must be 2-tuple or Box or rasterio Window '
                            'or shapely geometry.')

    @abstractmethod
    def _map_to_pixel(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Transform point from map to pixel coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates (eg. lon/lat). x and y
            can be single values or array-like.

        Returns:
            Tuple[int, int]: (x, y) tuple in pixel coordinates.
        """
        pass

    @abstractmethod
    def _pixel_to_map(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform point from pixel to map coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates. x and y can be
            single values or array-like.

        Returns:
            Tuple[float, float]: (x, y) tuple in map coordinates (eg. lon/lat).
        """
        pass
