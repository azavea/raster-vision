from abc import ABC, abstractmethod
from typing import Any, overload

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
                 transform: Any | None = None,
                 image_crs: str | None = None,
                 map_crs: str | None = None):
        self.transform = transform
        self.image_crs = image_crs
        self.map_crs = map_crs

    @overload
    def map_to_pixel(self, inp: tuple[float, float],
                     bbox: Box | None = None) -> tuple[int, int]:
        ...

    @overload
    def map_to_pixel(self,
                     inp: tuple['np.ndarray', 'np.ndarray'],
                     bbox: Box | None = None
                     ) -> tuple['np.ndarray', 'np.ndarray']:
        ...

    @overload
    def map_to_pixel(self, inp: Box, bbox: Box | None = None) -> Box:
        ...

    @overload
    def map_to_pixel(self, inp: BaseGeometry,
                     bbox: Box | None = None) -> BaseGeometry:
        ...

    def map_to_pixel(self, inp, bbox: Box | None = None):
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
            box_out = self._map_to_pixel_box(inp)
            if bbox is not None:
                box_out = box_out.to_local_coords(bbox)
            return box_out
        elif isinstance(inp, BaseGeometry):
            geom_out = self._map_to_pixel_geom(inp)
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                geom_out = translate(geom_out, xoff=-xmin, yoff=-ymin)
            return geom_out
        elif len(inp) == 2:
            out = self._map_to_pixel_point(inp)
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
    def pixel_to_map(self, inp: tuple[float, float],
                     bbox: Box | None = None) -> tuple[float, float]:
        ...

    @overload
    def pixel_to_map(self,
                     inp: tuple['np.ndarray', 'np.ndarray'],
                     bbox: Box | None = None
                     ) -> tuple['np.ndarray', 'np.ndarray']:
        ...

    @overload
    def pixel_to_map(self, inp: Box, bbox: Box | None = None) -> Box:
        ...

    @overload
    def pixel_to_map(self, inp: BaseGeometry,
                     bbox: Box | None = None) -> BaseGeometry:
        ...

    def pixel_to_map(self, inp, bbox: Box | None = None):
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
            box_out = self._pixel_to_map_box(box_in)
            return box_out
        elif isinstance(inp, BaseGeometry):
            geom_in = inp
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                geom_in = translate(geom_in, xoff=xmin, yoff=ymin)
            geom_out = self._pixel_to_map_geom(geom_in)
            return geom_out
        elif len(inp) == 2:
            if bbox is not None:
                xmin, ymin = bbox.xmin, bbox.ymin
                inp_x, inp_y = inp
                inp = (inp_x + xmin, inp_y + ymin)
            out = self._pixel_to_map_point(inp)
            out_x, out_y = out
            out = (np.array(out_x), np.array(out_y))
            return out
        else:
            raise TypeError(
                'Input must be 2-tuple or Box or shapely geometry.')

    @abstractmethod
    def _map_to_pixel_point(self,
                            point: tuple[float, float]) -> tuple[int, int]:
        """Transform point(s) from map to pixel coordinates.

        Args:
            map_point: ``(x, y)`` tuple in map coordinates (eg. lon/lat). ``x``
                and ``y`` can be single values or array-like.

        Returns:
            ``(x, y)`` tuple in pixel coordinates.
        """

    def _map_to_pixel_box(self, box: Box) -> Box:
        """Transform a :class:`Box` from map to pixel coordinates.

        Args:
            box: Box to transform.

        Returns:
            Box in pixel coordinates.
        """
        ymin, xmin, ymax, xmax = box
        xmin_tf, ymin_tf = self._map_to_pixel_point((xmin, ymin))
        xmax_tf, ymax_tf = self._map_to_pixel_point((xmax, ymax))
        pixel_box = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
        return pixel_box

    def _map_to_pixel_geom(self, geom: Box) -> Box:
        """Transform a shapely geom from map to pixel coordinates.

        Args:
            geom: Geom to transform.

        Returns:
            Geom in pixel coordinates.
        """
        pixel_geom = transform(
            lambda x, y, z=None: self._map_to_pixel_point((x, y)),
            geom,
        )
        return pixel_geom

    @abstractmethod
    def _pixel_to_map_point(self,
                            point: tuple[int, int]) -> tuple[float, float]:
        """Transform point(s) from pixel to map coordinates.

        Args:
            pixel_point: ``(x, y)`` tuple in pixel coordinates. ``x`` and ``y``
                can be single values or array-like.

        Returns:
            ``(x, y)`` tuple in map coordinates (eg. lon/lat).
        """

    def _pixel_to_map_box(self, box: Box) -> Box:
        """Transform a :class:`Box` from pixel to map coordinates.

        Args:
            box: Box to transform.

        Returns:
            Box in map coordinates (eg. lon/lat).
        """
        ymin, xmin, ymax, xmax = box
        xmin_tf, ymin_tf = self._pixel_to_map_point((xmin, ymin))
        xmax_tf, ymax_tf = self._pixel_to_map_point((xmax, ymax))
        map_box = Box(ymin_tf, xmin_tf, ymax_tf, xmax_tf)
        return map_box

    def _pixel_to_map_geom(self, geom: Box) -> Box:
        """Transform a shapely geom from pixel to map coordinates.

        Args:
            geom: Geom to transform.

        Returns:
            Geom in map coordinates.
        """
        map_geom = transform(
            lambda x, y, z=None: self._pixel_to_map_point((x, y)),
            geom,
        )
        return map_geom
