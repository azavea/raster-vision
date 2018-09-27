class CRSTransformer(object):
    """Transforms map points in some CRS into pixel coordinates.

    Each transformer is associated with a particular RasterSource."""

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
