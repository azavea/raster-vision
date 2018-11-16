import math
import random

import numpy as np
from shapely.geometry import box as ShapelyBox


class BoxSizeError(ValueError):
    pass


class Box():
    """A multi-purpose box (ie. rectangle)."""

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
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def __eq__(self, other):
        """Return true if other has same coordinates."""
        return self.tuple_format() == other.tuple_format()

    def __ne__(self, other):
        """Return true if other has different coordinates."""
        return self.tuple_format() != other.tuple_format()

    def get_height(self):
        """Return height of Box."""
        return self.ymax - self.ymin

    def get_width(self):
        """Return width of Box."""
        return self.xmax - self.xmin

    def get_area(self):
        """Return area of Box."""
        return self.get_height() * self.get_width()

    def rasterio_format(self):
        """Return Box in Rasterio format."""
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def tuple_format(self):
        return (self.ymin, self.xmin, self.ymax, self.xmax)

    def shapely_format(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def to_int(self):
        return Box(
            int(self.ymin), int(self.xmin), int(self.ymax), int(self.xmax))

    def npbox_format(self):
        """Return Box in npbox format used by TF Object Detection API.

        Returns:
            Numpy array of form [ymin, xmin, ymax, xmax] with float type

        """
        return np.array(
            [self.ymin, self.xmin, self.ymax, self.xmax], dtype=np.float)

    @staticmethod
    def to_npboxes(boxes):
        """Return nx4 numpy array from list of Box."""
        nb_boxes = len(boxes)
        npboxes = np.empty((nb_boxes, 4))
        for boxind, box in enumerate(boxes):
            npboxes[boxind, :] = box.npbox_format()
        return npboxes

    def __str__(self):  # pragma: no cover
        return str(self.npbox_format())

    def __repr__(self):  # pragma: no cover
        return str(self)

    def geojson_coordinates(self):
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
        if size < self.get_width():
            raise BoxSizeError('size of random container cannot be < width')

        if size < self.get_height():  # pragma: no cover
            raise BoxSizeError('size of random container cannot be < height')

        lb = self.ymin - (size - self.get_height())
        ub = self.ymin
        rand_y = random.randint(lb, ub)

        lb = self.xmin - (size - self.get_width())
        ub = self.xmin
        rand_x = random.randint(lb, ub)

        return Box.make_square(rand_y, rand_x, size)

    def make_random_square(self, size):
        """Return new randomly positioned square Box that lies inside this Box.

        Args:
            size: the height and width of the new Box

        """
        if size >= self.get_width():
            raise BoxSizeError('size of random square cannot be >= width')

        if size >= self.get_height():  # pragma: no cover
            raise BoxSizeError('size of random square cannot be >= height')

        lb = self.ymin
        ub = self.ymax - size
        rand_y = random.randint(lb, ub)

        lb = self.xmin
        ub = self.xmax - size
        rand_x = random.randint(lb, ub)

        return Box.make_square(rand_y, rand_x, size)

    def intersection(self, other):
        """Return the intersection of this Box and the other.

        Args:
            other: The box to intersect with this one.

        Returns:
             The intersection of this box and the other one.

        """
        xmin = max(self.xmin, other.xmin)
        ymin = max(self.ymin, other.ymin)
        xmax = min(self.xmax, other.xmax)
        ymax = min(self.ymax, other.ymax)
        return Box(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

    @staticmethod
    def from_npbox(npbox):
        """Return new Box based on npbox format.

        Args:
            npbox: Numpy array of form [ymin, xmin, ymax, xmax] with float type

        """
        return Box(*npbox)

    @staticmethod
    def from_shapely(shape):
        bounds = shape.bounds
        return Box(bounds[1], bounds[0], bounds[3], bounds[2])

    @staticmethod
    def from_tuple(tup):
        """Return new Box based on tuple format.

        Args:
           tup: Tuple format box (ymin, xmin, ymax, xmax)
        """
        return Box(tup[0], tup[1], tup[2], tup[3])

    def to_shapely(self):
        return ShapelyBox(*(self.shapely_format()))

    def reproject(self, transform_fn):
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
    def make_square(ymin, xmin, size):
        """Return new square Box."""
        return Box(ymin, xmin, ymin + size, xmin + size)

    def make_eroded(self, erosion_size):
        """Return new Box whose sides are eroded by erosion_size."""
        return Box(self.ymin + erosion_size, self.xmin + erosion_size,
                   self.ymax - erosion_size, self.xmax - erosion_size)

    def make_buffer(self, buffer_size, max_extent):
        """Return new Box whose sides are buffered by buffer_size.

        The resulting box is clipped so that the values of the corners are
        always greater than zero and less than the height and width of
        max_extent.

        """
        buffer_size = max(0., buffer_size)
        if buffer_size < 1.:
            delta_width = int(round(buffer_size * self.get_width()))
            delta_height = int(round(buffer_size * self.get_height()))
        else:
            delta_height = delta_width = int(round(buffer_size))

        return Box(
            max(0, math.floor(self.ymin - delta_height)),
            max(0, math.floor(self.xmin - delta_width)),
            min(max_extent.get_height(),
                int(self.ymax) + delta_height),
            min(max_extent.get_width(),
                int(self.xmax) + delta_width))

    def make_copy(self):
        return Box(*(self.tuple_format()))

    def get_windows(self, chip_size, stride):
        """Return list of grid of boxes within this box.

        Args:
            chip_size: (int) the length of each square-shaped window
            stride: (int) how much each window is offset from the last

        """
        height = self.get_height()
        width = self.get_width()

        result = []
        for row_start in range(0, height, stride):
            for col_start in range(0, width, stride):
                result.append(Box.make_square(row_start, col_start, chip_size))
        return result

    def to_dict(self):
        return {
            'xmin': self.xmin,
            'ymin': self.ymin,
            'xmax': self.xmax,
            'ymax': self.ymax
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['ymin'], d['xmin'], d['ymax'], d['xmax'])

    @staticmethod
    def filter_by_aoi(windows, aoi_polygons):
        """Filters windows by a list of AOI polygons"""
        result = []
        for window in windows:
            w = window.to_shapely()
            for polygon in aoi_polygons:
                if w.within(polygon):
                    result.append(window)
                    break

        return result
