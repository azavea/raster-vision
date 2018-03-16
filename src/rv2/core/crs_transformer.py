from abc import ABC, abstractmethod


class CRSTransformer(object):
    """Converts points between CRSs for a RasterSource."""

    # TODO did i get the ordering right in the docs?
    def web_to_pixel(self, web_point):
        """Return point in pixel coordinates.

        Args:
            web_point: tuple (long, lat) in WebMercator coordinates

        Returns:
            tuple (row, col) in pixel coordinates
        """
        pass

    def pixel_to_web(self, pixel_point):
        """Return point in Web Mercator coordinates.

        Args:
            pixel_point: tuple (row, col) in pixel coordinates

        Returns:
            tuple (long, lat) in WebMercator coordinates
        """
        pass
