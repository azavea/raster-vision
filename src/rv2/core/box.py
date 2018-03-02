import numpy as np


class Box():
    def __init__(self, ymin, xmin, ymax, xmax):
        self.ymin = ymin
        self.xmin = xmin
        self.ymax = ymax
        self.xmax = xmax

    def get_height(self):
        return self.ymax - self.ymin

    def get_width(self):
        return self.xmax - self.xmin

    def rasterio_format(self):
        return ((self.ymin, self.ymax), (self.xmin, self.xmax))

    def rtree_format(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def npbox_format(self):
        return np.array(
            [self.ymin, self.xmin, self.ymax, self.xmax], dtype=np.float)

    def __str__(self):
        return str(self.npbox_format())

    def __repr__(self):
        return str(self)

    def geojson_polygon_format(self):
        # Cardinal directions
        nw = (self.ymin, self.xmin)
        ne = (self.ymin, self.xmax)
        se = (self.ymax, self.xmax)
        sw = (self.ymax, self.xmin)
        return [nw, ne, se, sw, nw]

    def make_random_square_container(self, xlimit, ylimit, size):
        """Make random square window that contains box."""
        lb = max(0, self.ymin - (size - self.get_height()))
        ub = min(ylimit - size, self.ymin)
        rand_y = int(np.random.uniform(lb, ub))

        lb = max(0, self.xmin - (size - self.get_width()))
        ub = min(xlimit - size, self.xmin)
        rand_x = int(np.random.uniform(lb, ub))

        return Box.make_square(rand_y, rand_x, size)

    def make_random_square(self, size):
        """Make random square inside of box."""
        ub = self.get_height() - size
        rand_y = int(np.random.uniform(0, ub))

        ub = self.get_width() - size
        rand_x = int(np.random.uniform(0, ub))

        return Box.make_square(rand_y, rand_x, size)

    @staticmethod
    def from_npbox(npbox):
        return Box(*npbox)

    @staticmethod
    def make_square(ymin, xmin, size):
        return Box(ymin, xmin, ymin+size, xmin+size)
