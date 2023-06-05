from abc import ABC, abstractmethod
from typing import Any, Optional, overload, Tuple

import numpy as np
from shapely.ops import transform
from shapely.geometry.base import BaseGeometry
from shapely.affinity import translate

from rastervision.core.box import Box


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
    def map_to_pixel(self,
                     inp: Tuple[float, float],
                     bbox: Optional[Box] = None) -> Tuple[int, int]:
        ...

    @overload
    def map_to_pixel(self,
                     inp: Tuple['np.array', 'np.array'],
                     bbox: Optional[Box] = None
                     ) -> Tuple['np.array', 'np.array']:
        ...

    @overload
    def map_to_pixel(self, inp: Box, bbox: Optional[Box] = None) -> Box:
        ...

    @overload
    def map_to_pixel(self, inp: BaseGeometry,
                     bbox: Optional[Box] = None) -> BaseGeometry:
        ...

    def map_to_pixel(self, inp, bbox: Optional[Box] = None):
        """Transform input from map to pixel coords.

        Args:
            inp: (x, y) tuple or Box or shapely geometry in map coordinates.
                If tuple, x and y can be single values or array-like.
            bbox: If the extent of the associated RasterSource is constrained
                via a bbox, it can be passed here to get an output that is in
                coordinates of the bbox rather than the full extent of the data
                source of the RasterSource. Defaults to None.

        Returns:
            Coordinate-transformed input in the same format.
        """
        if isinstance(inp, Box):
            box_in = inp
            ymin, xmin, ymax, xmax = box_in
            xmin_tf, ymin_tf = self._map_to_pixel((xmin, ymin))
            xmax_tf, ymax_tf = self._map_to_pixel((xmax, ymax))
            box_out = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
            if bbox is not None:
                box_out = box_out.to_local_coords(bbox)
            return box_out
        elif isinstance(inp, BaseGeometry):
            geom_in = inp
            geom_out = transform(
                lambda x, y, z=None: self._map_to_pixel((x, y)), geom_in)
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                geom_out = translate(geom_out, xoff=-xmin, yoff=-ymin)
            return geom_out
        elif len(inp) == 2:
            out = self._map_to_pixel(inp)
            out_x, out_y = out
            out = (np.array(out_x), np.array(out_y))
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                out_x, out_y = out
                out = (out_x - xmin, out_y - ymin)
            return out
        else:
            raise TypeError(
                'Input must be 2-tuple or Box or shapely geometry.')

    @overload
    def pixel_to_map(self,
                     inp: Tuple[float, float],
                     bbox: Optional[Box] = None) -> Tuple[float, float]:
        ...

    @overload
    def pixel_to_map(self,
                     inp: Tuple['np.array', 'np.array'],
                     bbox: Optional[Box] = None
                     ) -> Tuple['np.array', 'np.array']:
        ...

    @overload
    def pixel_to_map(self, inp: Box, bbox: Optional[Box] = None) -> Box:
        ...

    @overload
    def pixel_to_map(self, inp: BaseGeometry,
                     bbox: Optional[Box] = None) -> BaseGeometry:
        ...

    def pixel_to_map(self, inp, bbox: Optional[Box] = None):
        """Transform input from pixel to map coords.

        Args:
            inp: (x, y) tuple or Box or shapely geometry in pixel coordinates.
                If tuple, x and y can be single values or array-like.
            bbox: If the extent of the associated RasterSource is constrained
                via a bbox, it can be passed here so that the input is
                interpreted to be in coordinates of the bbox rather than the
                full extent of the data source of the RasterSource.
                Defaults to None.

        Returns:
            Coordinate-transformed input in the same format.
        """
        if isinstance(inp, Box):
            box_in = inp
            if bbox is not None:
                box_in = box_in.to_global_coords(bbox)
            ymin, xmin, ymax, xmax = box_in
            xmin_tf, ymin_tf = self._pixel_to_map((xmin, ymin))
            xmax_tf, ymax_tf = self._pixel_to_map((xmax, ymax))
            box_out = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
            return box_out
        elif isinstance(inp, BaseGeometry):
            geom_in = inp
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                geom_in = translate(geom_in, xoff=xmin, yoff=ymin)
            geom_out = transform(
                lambda x, y, z=None: self._pixel_to_map((x, y)), geom_in)
            return geom_out
        elif len(inp) == 2:
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                inp_x, inp_y = inp
                inp = (inp_x + xmin, inp_y + ymin)
            out = self._pixel_to_map(inp)
            out_x, out_y = out
            out = (np.array(out_x), np.array(out_y))
            return out
        else:
            raise TypeError(
                'Input must be 2-tuple or Box or shapely geometry.')

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
