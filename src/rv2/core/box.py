import numpy as np


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

    def get_height(self):
        """Return height of Box."""
        return self.ymax - self.ymin

    def get_width(self):
        """Return width of Box."""
        return self.xmax - self.xmin

    def rasterio_format(self):
        """Return Box in Rasterio format."""
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def npbox_format(self):
        """Return Box in npbox format used by TF Object Detection API.

        Returns:
            Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return np.array(
            [self.ymin, self.xmin, self.ymax, self.xmax], dtype=np.float)

    def __str__(self):
        return str(self.npbox_format())

    def __repr__(self):
        return str(self)

    def geojson_coordinates(self):
        """Return Box as GeoJSON coordinates.
        """
        # Compass directions:
        nw = (self.xmin, self.ymin)
        ne = (self.xmin, self.ymax)
        se = (self.xmax, self.ymax)
        sw = (self.xmax, self.ymin)
        return [nw, ne, se, sw, nw]

    def make_random_square_container(self, xlimit, ylimit, size):
        """Return a new square Box that contains this Box.

        Assumes that the minimum x and y values are 0.

        Args:
            xlimit: the maximum x value for the new Box
            ylimit: the maximum y value for the new Box
            size: the width and height of the new Box
        """
        lb = max(0, self.ymin - (size - self.get_height()))
        ub = min(ylimit - size, self.ymin)
        rand_y = int(np.random.uniform(lb, ub))

        lb = max(0, self.xmin - (size - self.get_width()))
        ub = min(xlimit - size, self.xmin)
        rand_x = int(np.random.uniform(lb, ub))

        return Box.make_square(rand_y, rand_x, size)

    def make_random_square(self, size):
        """Return new randomly positioned square Box that lies inside this Box.

        Args:
            size: the height and width of the new Box
        """
        ub = self.get_height() - size
        rand_y = int(np.random.uniform(0, ub))

        ub = self.get_width() - size
        rand_x = int(np.random.uniform(0, ub))

        return Box.make_square(rand_y, rand_x, size)

    @staticmethod
    def from_npbox(npbox):
        """Return new Box based on npbox format.

        Args:
            npbox: Numpy array of form [ymin, xmin, ymax, xmax] with float type
        """
        return Box(*npbox)

    @staticmethod
    def make_square(ymin, xmin, size):
        """Return new square Box."""
        return Box(ymin, xmin, ymin+size, xmin+size)
