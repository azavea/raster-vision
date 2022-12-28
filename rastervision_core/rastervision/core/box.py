from typing import TYPE_CHECKING, Callable, Dict, Union, Tuple, Optional, List
from typing_extensions import Literal
from pydantic import PositiveInt as PosInt, conint
import math
import random

import numpy as np
from shapely.geometry import Polygon
from rasterio.windows import Window as RioWindow

NonNegInt = conint(ge=0)

if TYPE_CHECKING:
    pass


class BoxSizeError(ValueError):
    pass


class Box():
    """A multi-purpose box (ie. rectangle) representation ."""

    def __init__(self, ymin, xmin, ymax, xmax):
        """Construct a bounding box.

        Unless otherwise stated, the convention is that these coordinates are
        in pixel coordinates and represent boxes that lie within a
        RasterSource.

        Args:
            ymin: minimum y value (y is row)
            xmin: minimum x value (x is column)
            ymax: maximum y value
            xmax: maximum x value

        """
        ymin, ymax = sorted((ymin, ymax))
        xmin, xmax = sorted((xmin, xmax))
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def __eq__(self, other: 'Box') -> bool:
        """Return true if other has same coordinates."""
        return self.tuple_format() == other.tuple_format()

    def __ne__(self, other: 'Box'):
        """Return true if other has different coordinates."""
        return self.tuple_format() != other.tuple_format()

    @property
    def height(self) -> int:
        """Return height of Box."""
        return self.ymax - self.ymin

    @property
    def width(self) -> int:
        """Return width of Box."""
        return self.xmax - self.xmin

    @property
    def size(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def area(self) -> int:
        """Return area of Box."""
        return self.height * self.width

    def rasterio_format(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return Box in Rasterio format: ((ymin, ymax), (xmin, xmax))."""
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def tuple_format(self) -> Tuple[int, int, int, int]:
        return (self.ymin, self.xmin, self.ymax, self.xmax)

    def shapely_format(self) -> Tuple[int, int, int, int]:
        return self.to_xyxy()

    def to_int(self):
        return Box(
            int(self.ymin), int(self.xmin), int(self.ymax), int(self.xmax))

    def npbox_format(self):
        """Return Box in npbox format used by TF Object Detection API.

        Returns:
            Numpy array of form [ymin, xmin, ymax, xmax] with float type

        """
        return np.array(
            [self.ymin, self.xmin, self.ymax, self.xmax], dtype=float)

    @staticmethod
    def to_npboxes(boxes):
        """Return nx4 numpy array from list of Box."""
        nb_boxes = len(boxes)
        npboxes = np.empty((nb_boxes, 4))
        for boxind, box in enumerate(boxes):
            npboxes[boxind, :] = box.npbox_format()
        return npboxes

    def __iter__(self):
        return iter(self.tuple_format())

    def __getitem__(self, i):
        return self.tuple_format()[i]

    def __repr__(self) -> str:
        arg_keys = ['ymin', 'xmin', 'ymax', 'xmax']
        arg_vals = [getattr(self, k) for k in arg_keys]
        arg_strs = [f'{k}={v}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def __hash__(self) -> int:
        return hash(self.tuple_format())

    def geojson_coordinates(self) -> List[Tuple[int, int]]:
        """Return Box as GeoJSON coordinates."""
        # Compass directions:
        nw = [self.xmin, self.ymin]
        ne = [self.xmin, self.ymax]
        se = [self.xmax, self.ymax]
        sw = [self.xmax, self.ymin]
        return [nw, ne, se, sw, nw]

    def make_random_square_container(self, size):
        """Return a new square Box that contains this Box.

        Args:
            size: the width and height of the new Box

        """
        if size < self.width:
            raise BoxSizeError('size of random container cannot be < width')

        if size < self.height:  # pragma: no cover
            raise BoxSizeError('size of random container cannot be < height')

        lb = self.ymin - (size - self.height)
        ub = self.ymin
        rand_y = random.randint(int(lb), int(ub))

        lb = self.xmin - (size - self.width)
        ub = self.xmin
        rand_x = random.randint(int(lb), int(ub))

        return Box.make_square(rand_y, rand_x, size)

    def make_random_box_container(self, out_h: int, out_w: int) -> 'Box':
        """Return a new rectangular Box that contains this Box.

        Args:
            out_h (int): the height of the new Box
            out_w (int): the width of the new Box
        """
        self_h, self_w = self.size

        if out_h < self_h:  # pragma: no cover
            raise BoxSizeError('size of random container cannot be < height')
        if out_w < self_w:
            raise BoxSizeError('size of random container cannot be < width')

        lb = self.ymin - (out_h - self_h)
        ub = self.ymin
        ymin = random.randint(int(lb), int(ub))

        lb = self.xmin - (out_w - self_w)
        ub = self.xmin
        xmin = random.randint(int(lb), int(ub))

        return Box(ymin, xmin, ymin + out_h, xmin + out_w)

    def make_random_square(self, size: int) -> 'Box':
        """Return new randomly positioned square Box that lies inside this Box.

        Args:
            size: the height and width of the new Box

        """
        if size >= self.width:
            raise BoxSizeError('size of random square cannot be >= width')

        if size >= self.height:  # pragma: no cover
            raise BoxSizeError('size of random square cannot be >= height')

        lb = self.ymin
        ub = self.ymax - size
        rand_y = random.randint(int(lb), int(ub))

        lb = self.xmin
        ub = self.xmax - size
        rand_x = random.randint(int(lb), int(ub))

        return Box.make_square(rand_y, rand_x, size)

    def intersection(self, other: 'Box') -> 'Box':
        """Return the intersection of this Box and the other.

        Args:
            other: The box to intersect with this one.

        Returns:
             The intersection of this box and the other one.

        """
        if not self.intersects(other):
            return Box(0, 0, 0, 0)
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    def intersects(self, other: 'Box') -> bool:
        if self.ymax <= other.ymin or self.ymin >= other.ymax:
            return False
        if self.xmax <= other.xmin or self.xmin >= other.xmax:
            return False
        return True

    @staticmethod
    def from_npbox(npbox):
        """Return new Box based on npbox format.

        Args:
            npbox: Numpy array of form [ymin, xmin, ymax, xmax] with float type

        """
        return Box(*npbox)

    @staticmethod
    def from_shapely(shape):
        xmin, ymin, xmax, ymax = shape.bounds
        return Box(ymin, xmin, ymax, xmax)

    @classmethod
    def from_rasterio(self, rio_window: RioWindow) -> 'Box':
        yslice, xslice = rio_window.toslices()
        return Box(yslice.start, xslice.start, yslice.stop, xslice.stop)

    def to_xywh(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.width, self.height)

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_points(self) -> np.ndarray:
        """Get (x, y) coords of each vertex as a 4x2 numpy array."""
        return np.array(self.geojson_coordinates()[:4])

    def to_shapely(self) -> Polygon:
        """Convert to shapely Polygon."""
        return Polygon.from_bounds(*(self.shapely_format()))

    def to_rasterio(self) -> RioWindow:
        """Convert to a Rasterio Window."""
        return RioWindow.from_slices(*self.to_slices())

    def to_slices(self) -> Tuple[slice, slice]:
        """Convert to slices: ymin:ymax, xmin:xmax"""
        return slice(self.ymin, self.ymax), slice(self.xmin, self.xmax)

    def translate(self, dy: int, dx: int) -> 'Box':
        """Translate window along y and x axes by the given distances."""
        ymin, xmin, ymax, xmax = self
        return Box(ymin + dy, xmin + dx, ymax + dy, xmax + dx)

    def shift_origin(self, extent: 'Box') -> 'Box':
        """Shift origin of window coords to (extent.xmin, extent.ymin)."""
        return self.translate(dy=extent.ymin, dx=extent.xmin)

    def to_offsets(self, container: 'Box') -> 'Box':
        """Convert coords to offsets from (container.xmin, container.ymin)."""
        return self.translate(dy=-container.ymin, dx=-container.xmin)

    def reproject(self, transform_fn: Callable) -> 'Box':
        """Reprojects this box based on a transform function.

        Args:
          transform_fn - A function that takes in a tuple (x, y)
                         and reprojects that point to the target
                         coordinate reference system.
        """
        (xmin, ymin) = transform_fn((self.xmin, self.ymin))
        (xmax, ymax) = transform_fn((self.xmax, self.ymax))

        return Box(ymin, xmin, ymax, xmax)

    @staticmethod
    def make_square(ymin, xmin, size) -> 'Box':
        """Return new square Box."""
        return Box(ymin, xmin, ymin + size, xmin + size)

    def center_crop(self, edge_offset_y: int, edge_offset_x: int) -> 'Box':
        """Return Box whose sides are eroded by the given offsets.

        Box(0, 0, 10, 10).center_crop(2, 4) ==  Box(2, 4, 8, 6)
        """
        return Box(self.ymin + edge_offset_y, self.xmin + edge_offset_x,
                   self.ymax - edge_offset_y, self.xmax - edge_offset_x)

    def erode(self, erosion_sz) -> 'Box':
        """Return new Box whose sides are eroded by erosion_sz."""
        return self.center_crop(erosion_sz, erosion_sz)

    def buffer(self, buffer_sz: float, max_extent: 'Box') -> 'Box':
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

    def pad(self, ymin: int, xmin: int, ymax: int, xmax: int) -> 'Box':
        """Pad sides by the given amount."""
        return Box(
            ymin=self.ymin - ymin,
            xmin=self.xmin - xmin,
            ymax=self.ymax + ymax,
            xmax=self.xmax + xmax)

    def copy(self) -> 'Box':
        return Box(*self)

    def get_windows(self,
                    size: Union[PosInt, Tuple[PosInt, PosInt]],
                    stride: Union[PosInt, Tuple[PosInt, PosInt]],
                    padding: Optional[Union[NonNegInt, Tuple[
                        NonNegInt, NonNegInt]]] = None,
                    pad_direction: Literal['both', 'start', 'end'] = 'end'
                    ) -> List['Box']:
        """Returns a list of boxes representing windows generated using a
        sliding window traversal with the specified size, stride, and
        padding.

        Each of size, stride, and padding can be either a positive int or
        a tuple `(vertical-componet, horizontal-component)` of positive ints.

        Padding currently only applies to the right and bottom edges.

        Args:
            size (Union[PosInt, Tuple[PosInt, PosInt]]): Size (h, w) of the
                windows.
            stride (Union[PosInt, Tuple[PosInt, PosInt]]): Step size between
                windows. Can be 2-tuple (h_step, w_step) or positive int.
            padding (Optional[Union[PosInt, Tuple[PosInt, PosInt]]], optional):
                Optional padding to accomodate windows that overflow the
                extent. Can be 2-tuple (h_pad, w_pad) or non-negative int.
                If None, will be set to (size[0]//2, size[1]//2).
                Defaults to None.
            pad_direction (Literal['both', 'start', 'end']): If 'end', only pad
                ymax and xmax (bottom and right). If 'start', only pad ymin and
                xmin (top and left). If 'both', pad all sides. Has no effect if
                paddiong is zero. Defaults to 'end'.

        Returns:
            List[Box]: List of Box objects.
        """
        if not isinstance(size, tuple):
            size = (size, size)

        if not isinstance(stride, tuple):
            stride = (stride, stride)

        if size[0] <= 0 or size[1] <= 0 or stride[0] <= 0 or stride[1] <= 0:
            raise ValueError('size and stride must be positive.')

        if padding is None:
            padding = (size[0] // 2, size[1] // 2)

        if not isinstance(padding, tuple):
            padding = (padding, padding)

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

    def to_dict(self) -> Dict[str, int]:
        return {
            'xmin': self.xmin,
            'ymin': self.ymin,
            'xmax': self.xmax,
            'ymax': self.ymax
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Box':
        return cls(d['ymin'], d['xmin'], d['ymax'], d['xmax'])

    @staticmethod
    def filter_by_aoi(windows: List['Box'],
                      aoi_polygons: List[Polygon],
                      within: bool = True) -> List['Box']:
        """Filters windows by a list of AOI polygons

        Args:
            within: if True, windows are only kept if they lie fully within an
                AOI polygon. Otherwise, windows are kept if they intersect an AOI
                polygon.
        """
        result = []
        for window in windows:
            w = window.to_shapely()
            for polygon in aoi_polygons:
                if ((within and w.within(polygon))
                        or ((not within) and w.intersects(polygon))):
                    result.append(window)
                    break

        return result

    @staticmethod
    def within_aoi(window: 'Box', aoi_polygons: List[Polygon]) -> bool:
        """Check if window is within a list of AOI polygons."""
        w = window.to_shapely()
        for polygon in aoi_polygons:
            if w.within(polygon):
                return True
        return False
