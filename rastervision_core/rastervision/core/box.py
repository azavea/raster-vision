from typing import (TYPE_CHECKING, Literal)
from collections.abc import Callable
from pydantic import NonNegativeInt as NonNegInt, PositiveInt as PosInt
import math
import random

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from rasterio.windows import Window as RioWindow

from rastervision.pipeline.utils import repr_with_args
from rastervision.core.utils.misc import (calculate_required_padding,
                                          ensure_tuple)

if TYPE_CHECKING:
    from typing import Self
    from shapely.geometry import MultiPolygon
    from shapely.geometry.base import BaseGeometry


class BoxSizeError(ValueError):
    pass


class Box:
    """A multi-purpose box (ie. rectangle) representation."""

    def __init__(self, ymin: int, xmin: int, ymax: int, xmax: int):
        """Constructor.

        Although primarily intended for representing integer pixel coordinates
        in a scene, this class can also be used to represent floating point
        map coordinates though not all methods might be compatible with that
        interpretation.

        Args:
            ymin: minimum y value (y is row)
            xmin: minimum x value (x is column)
            ymax: maximum y value
            xmax: maximum x value
        """
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def __eq__(self, other: 'Self') -> bool:
        """Return true if other has same coordinates."""
        return self.tuple_format() == other.tuple_format()

    def __ne__(self, other: 'Self'):
        """Return true if other has different coordinates."""
        return self.tuple_format() != other.tuple_format()

    @property
    def height(self) -> int:
        """Height of the Box."""
        return self.ymax - self.ymin

    @property
    def width(self) -> int:
        """Width of the Box."""
        return self.xmax - self.xmin

    @property
    def extent(self) -> 'Self':
        """Return a Box(0, 0, h, w) representing the size of this Box."""
        return Box(0, 0, self.height, self.width)

    @property
    def size(self) -> tuple[int, int]:
        """(height, width) tuple."""
        return self.height, self.width

    @property
    def area(self) -> int:
        """Return area of Box."""
        return self.height * self.width

    def normalize(self) -> 'Self':
        """Ensure ymin <= ymax and xmin <= xmax."""
        ymin, ymax = sorted((self.ymin, self.ymax))
        xmin, xmax = sorted((self.xmin, self.xmax))
        return Box(ymin, xmin, ymax, xmax)

    def rasterio_format(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return Box in Rasterio format: ((ymin, ymax), (xmin, xmax))."""
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def tuple_format(self) -> tuple[int, int, int, int]:
        """Return a (ymin, xmin, ymax, xmax) tuple."""
        return (self.ymin, self.xmin, self.ymax, self.xmax)

    def shapely_format(self) -> tuple[int, int, int, int]:
        return self.to_xyxy()

    def to_int(self):
        """Return a new Box with all coordinates cast to ints."""
        ymin, xmin, ymax, xmax = self
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
        out = Box(ymin, xmin, ymax, xmax)
        return out

    def npbox_format(self) -> np.ndarray:
        """Return Box in npbox format used by TF Object Detection API.

        Returns:
            Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return np.array(self.tuple_format(), dtype=float)

    @staticmethod
    def to_npboxes(boxes: list['Self']) -> np.ndarray:
        """Return nx4 numpy array from list of Box."""
        nb_boxes = len(boxes)
        npboxes = np.empty((nb_boxes, 4))
        for boxind, box in enumerate(boxes):
            npboxes[boxind, :] = box.npbox_format()
        return npboxes

    def __iter__(self):
        return iter(self.tuple_format())

    def __getitem__(self, i: NonNegInt):
        return self.tuple_format()[i]

    def __repr__(self) -> str:
        return repr_with_args(self, **self.to_dict())

    def __hash__(self) -> int:
        return hash(self.tuple_format())

    def geojson_coordinates(self) -> list[tuple[int, int]]:
        """Return Box as GeoJSON coordinates."""
        # Compass directions:
        nw = [self.xmin, self.ymin]
        ne = [self.xmin, self.ymax]
        se = [self.xmax, self.ymax]
        sw = [self.xmax, self.ymin]
        return [nw, ne, se, sw, nw]

    def make_random_square_container(self, size: int) -> 'Self':
        """Return a new square Box that contains this Box.

        Args:
            size: the width and height of the new Box
        """
        return self.make_random_box_container(size, size)

    def make_random_box_container(self, out_h: int, out_w: int) -> 'Self':
        """Return a new rectangular Box that contains this Box.

        Args:
            out_h (int): the height of the new Box
            out_w (int): the width of the new Box
        """
        self_h, self_w = self.size

        if out_h < self_h:
            raise BoxSizeError('size of random container cannot be < height')
        if out_w < self_w:
            raise BoxSizeError('size of random container cannot be < width')

        ymin, xmin, _, _ = self.normalize()

        lb = ymin - (out_h - self_h)
        ub = ymin
        out_ymin = random.randint(int(lb), int(ub))

        lb = xmin - (out_w - self_w)
        ub = xmin
        out_xmin = random.randint(int(lb), int(ub))

        return Box(out_ymin, out_xmin, out_ymin + out_h, out_xmin + out_w)

    def make_random_square(self, size: int) -> 'Self':
        """Return new randomly positioned square Box that lies inside this Box.

        Args:
            size: the height and width of the new Box
        """
        if size >= self.width:
            raise BoxSizeError('size of random square cannot be >= width')

        if size >= self.height:
            raise BoxSizeError('size of random square cannot be >= height')

        ymin, xmin, ymax, xmax = self.normalize()

        lb = ymin
        ub = ymax - size
        rand_y = random.randint(int(lb), int(ub))

        lb = xmin
        ub = xmax - size
        rand_x = random.randint(int(lb), int(ub))

        return Box.make_square(rand_y, rand_x, size)

    def intersection(self, other: 'Self') -> 'Self':
        """Return the intersection of this Box and the other.

        Args:
            other: The box to intersect with this one.

        Returns:
             The intersection of this box and the other one.
        """
        if not self.intersects(other):
            return Box(0, 0, 0, 0)

        box1 = self.normalize()
        box2 = other.normalize()

        xmin = max(box1.xmin, box2.xmin)
        ymin = max(box1.ymin, box2.ymin)
        xmax = min(box1.xmax, box2.xmax)
        ymax = min(box1.ymax, box2.ymax)
        return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def intersects(self, other: 'Self') -> bool:
        box1 = self.normalize()
        box2 = other.normalize()
        if box1.ymax <= box2.ymin or box1.ymin >= box2.ymax:
            return False
        if box1.xmax <= box2.xmin or box1.xmin >= box2.xmax:
            return False
        return True

    @classmethod
    def from_npbox(cls, npbox: np.ndarray) -> 'Self':
        """Return new Box based on npbox format.

        Args:
            npbox: Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return Box(*npbox)

    @classmethod
    def from_shapely(cls, shape: 'BaseGeometry') -> 'Self':
        """Instantiate from the bounds of a shapely geometry."""
        xmin, ymin, xmax, ymax = shape.bounds
        return Box(ymin, xmin, ymax, xmax)

    @classmethod
    def from_rasterio(cls, rio_window: RioWindow) -> 'Self':
        """Instantiate from a rasterio window."""
        yslice, xslice = rio_window.toslices()
        return Box(yslice.start, xslice.start, yslice.stop, xslice.stop)

    def to_xywh(self) -> tuple[int, int, int, int]:
        """Convert to (xmin, ymin, width, height) tuple"""
        return (self.xmin, self.ymin, self.width, self.height)

    def to_xyxy(self) -> tuple[int, int, int, int]:
        """Convert to (xmin, ymin, xmax, ymax) tuple"""
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_points(self) -> np.ndarray:
        """Get (x, y) coords of each vertex as a 4x2 numpy array."""
        return np.array(self.geojson_coordinates()[:4])

    def to_shapely(self) -> Polygon:
        """Convert to shapely Polygon."""
        return Polygon.from_bounds(*self.shapely_format())

    def to_rasterio(self) -> RioWindow:
        """Convert to a Rasterio Window."""
        return RioWindow.from_slices(*self.normalize().to_slices())

    def to_slices(self, h_step: int | None = None,
                  w_step: int | None = None) -> tuple[slice, slice]:
        """Convert to slices: ymin:ymax[:h_step], xmin:xmax[:w_step]"""
        return slice(self.ymin, self.ymax, h_step), slice(
            self.xmin, self.xmax, w_step)

    def translate(self, dy: int, dx: int) -> 'Self':
        """Translate window along y and x axes by the given distances."""
        ymin, xmin, ymax, xmax = self
        return Box(ymin + dy, xmin + dx, ymax + dy, xmax + dx)

    def to_global_coords(self, bbox: 'Self') -> 'Self':
        """Go from bbox coords to global coords.

        E.g., Given a box Box(20, 20, 40, 40) and bbox Box(20, 20, 100, 100),
        the box becomes Box(40, 40, 60, 60).

        Inverse of Box.to_local_coords().
        """
        return self.translate(dy=bbox.ymin, dx=bbox.xmin)

    def to_local_coords(self, bbox: 'Self') -> 'Self':
        """Go from to global coords bbox coords.

        E.g., Given a box Box(40, 40, 60, 60) and bbox Box(20, 20, 100, 100),
        the box becomes Box(20, 20, 40, 40).

        Inverse of Box.to_global_coords().
        """
        return self.translate(dy=-bbox.ymin, dx=-bbox.xmin)

    def reproject(self, transform_fn: Callable[[tuple], tuple]) -> 'Self':
        """Reprojects this box based on a transform function.

        Args:
            transform_fn: A function that takes in a tuple (x, y) and
                reprojects that point to the target coordinate reference
                system.
        """
        (xmin, ymin) = transform_fn((self.xmin, self.ymin))
        (xmax, ymax) = transform_fn((self.xmax, self.ymax))

        return Box(ymin, xmin, ymax, xmax)

    @staticmethod
    def make_square(ymin, xmin, size) -> 'Self':
        """Return new square Box."""
        return Box(ymin, xmin, ymin + size, xmin + size)

    def center_crop(self, edge_offset_y: int, edge_offset_x: int) -> 'Self':
        """Return Box whose sides are eroded by the given offsets.

        Box(0, 0, 10, 10).center_crop(2, 4) ==  Box(2, 4, 8, 6)
        """
        return Box(self.ymin + edge_offset_y, self.xmin + edge_offset_x,
                   self.ymax - edge_offset_y, self.xmax - edge_offset_x)

    def erode(self, erosion_sz) -> 'Self':
        """Return new Box whose sides are eroded by erosion_sz."""
        return self.center_crop(erosion_sz, erosion_sz)

    def buffer(self, buffer_sz: float, max_extent: 'Self') -> 'Self':
        """Return new Box whose sides are buffered by buffer_sz.

        The resulting box is clipped so that the values of the corners are
        always greater than zero and less than the height and width of
        max_extent.
        """
        buffer_sz = max(0., buffer_sz)
        if buffer_sz < 1.:
            delta_width = int(round(buffer_sz * self.width))
            delta_height = int(round(buffer_sz * self.height))
        else:
            delta_height = delta_width = int(round(buffer_sz))

        return Box(
            max(0, math.floor(self.ymin - delta_height)),
            max(0, math.floor(self.xmin - delta_width)),
            min(max_extent.height,
                int(self.ymax) + delta_height),
            min(max_extent.width,
                int(self.xmax) + delta_width))

    def pad(self, ymin: int, xmin: int, ymax: int, xmax: int) -> 'Self':
        """Pad sides by the given amount."""
        return Box(
            ymin=self.ymin - ymin,
            xmin=self.xmin - xmin,
            ymax=self.ymax + ymax,
            xmax=self.xmax + xmax)

    def copy(self) -> 'Self':
        return Box(*self)

    def get_windows(
            self,
            size: PosInt | tuple[PosInt, PosInt],
            stride: PosInt | tuple[PosInt, PosInt],
            padding: NonNegInt | tuple[NonNegInt, NonNegInt] | None = None,
            pad_direction: Literal['both', 'start', 'end'] = 'end'
    ) -> list['Self']:
        """Return sliding windows for given size, stride, and padding.

        Each of size, stride, and padding can be either a positive int or
        a tuple ``(vertical-component, horizontal-component)`` of positive
        ints.

        If ``padding`` is not specified and ``stride <= size``, it will be
        automatically calculated such that the windows cover the entire extent.

        Args:
            size: Size (h, w) of the windows.
            stride: Step size between windows. Can be 2-tuple (h_step, w_step)
                or positive int.
            padding: Optional padding to accommodate windows that overflow the
                extent. Can be 2-tuple (h_pad, w_pad) or non-negative int.
                If None, will be automatically calculated such that the windows
                cover the entire extent. Defaults to ``None``.
            pad_direction: If ``'end'``, only pad ymax and xmax (bottom and
                right). If ``'start'``, only pad ymin and xmin (top and left).
                If ``'both'``, pad all sides. If ``'both'`` pad all sides. Has
                no effect if padding is zero. Defaults to ``'end'``.

        Returns:
            List of windows.
        """
        size: tuple[PosInt, PosInt] = ensure_tuple(size)
        stride: tuple[PosInt, PosInt] = ensure_tuple(stride)

        if size[0] <= 0 or size[1] <= 0 or stride[0] <= 0 or stride[1] <= 0:
            raise ValueError('size and stride must be positive.')

        if padding is None:
            if size[0] < stride[0] or size[1] < stride[1]:
                padding = (0, 0)
            else:
                padding = calculate_required_padding(self.size, size, stride,
                                                     pad_direction)

        padding: tuple[NonNegInt, NonNegInt] = ensure_tuple(padding)

        if padding[0] < 0 or padding[1] < 0:
            raise ValueError('padding must be non-negative.')

        if padding != (0, 0):
            h_pad, w_pad = padding
            if pad_direction == 'both':
                padded_box = self.pad(
                    ymin=h_pad, xmin=w_pad, ymax=h_pad, xmax=w_pad)
            elif pad_direction == 'end':
                padded_box = self.pad(ymin=0, xmin=0, ymax=h_pad, xmax=w_pad)
            elif pad_direction == 'start':
                padded_box = self.pad(ymin=h_pad, xmin=w_pad, ymax=0, xmax=0)
            else:
                raise ValueError('pad_directions must be one of: '
                                 '"both", "start", "end".')
            return padded_box.get_windows(
                size=size, stride=stride, padding=(0, 0))

        # padding is necessarily (0, 0) at this point, so we ignore it
        h, w = size
        h_step, w_step = stride
        # lb = lower bound, ub = upper bound
        ymin_lb = self.ymin
        xmin_lb = self.xmin
        ymin_ub = self.ymax - h
        xmin_ub = self.xmax - w

        windows = []
        for ymin in range(ymin_lb, ymin_ub + 1, h_step):
            for xmin in range(xmin_lb, xmin_ub + 1, w_step):
                windows.append(Box(ymin, xmin, ymin + h, xmin + w))
        return windows

    def to_dict(self) -> dict[str, int]:
        """Convert to a dict with keys: ymin, xmin, ymax, xmax."""
        return {
            'ymin': self.ymin,
            'xmin': self.xmin,
            'ymax': self.ymax,
            'xmax': self.xmax,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Self':
        return cls(d['ymin'], d['xmin'], d['ymax'], d['xmax'])

    @staticmethod
    def filter_by_aoi(windows: list['Self'],
                      aoi_polygons: list[Polygon],
                      within: bool = True) -> list['Self']:
        """Filters windows by a list of AOI polygons

        Args:
            within: if True, windows are only kept if they lie fully within an
                AOI polygon. Otherwise, windows are kept if they intersect an
                AOI polygon.
        """
        # merge overlapping polygons, if any
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)

        if within:
            keep_window = aoi_polygons.contains
        else:
            keep_window = aoi_polygons.intersects

        out = [w for w in windows if keep_window(w.to_shapely())]
        return out

    @staticmethod
    def within_aoi(window: 'Self',
                   aoi_polygons: Polygon | list[Polygon]) -> bool:
        """Check if window is within the union of given AOI polygons."""
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)
        w = window.to_shapely()
        out = aoi_polygons.contains(w)
        return out

    @staticmethod
    def intersects_aoi(window: 'Self',
                       aoi_polygons: Polygon | list[Polygon]) -> bool:
        """Check if window intersects with the union of given AOI polygons."""
        aoi_polygons: Polygon | MultiPolygon = unary_union(aoi_polygons)
        w = window.to_shapely()
        out = aoi_polygons.intersects(w)
        return out

    def __contains__(self, query: 'Self | tuple[int, int]') -> bool:
        """Check if box or point is contained within this box.

        Args:
            query: Box or single point (x, y).

        Raises:
            NotImplementedError: if query is not a Box or tuple/list.
        """
        if isinstance(query, Box):
            ymin, xmin, ymax, xmax = query
            return (ymin >= self.ymin and xmin >= self.xmin
                    and ymax <= self.ymax and xmax <= self.xmax)
        elif isinstance(query, (tuple, list)):
            x, y = query
            return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
        else:
            raise NotImplementedError()
