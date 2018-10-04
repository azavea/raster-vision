class CRSTransformer(object):
    """Transforms map points in some CRS into pixel coordinates.

    Each transformer is associated with a particular RasterSource."""

    def __init__(self, image_crs=None, map_crs=None):
        self.image_crs = image_crs
        self.map_crs = map_crs

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates (eg. lon/lat)

        Returns:
            (x, y) tuple in pixel coordinates
        """
        pass

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in map coordinates (eg. lon/lat)
        """
        pass

    def get_image_crs(self):
        return self.image_crs

    def get_map_crs(self):
        return self.map_crs
