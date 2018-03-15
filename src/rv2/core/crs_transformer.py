import pyproj


# TODO turn into abstract class and create subclass RasterioCRSTransformer
class CRSTransformer(object):
    """Converts points between CRSs for a RasterSource."""
    def __init__(self, image_dataset):
        self.image_dataset = image_dataset
        self.web_proj = pyproj.Proj(init='epsg:4326')
        image_crs = image_dataset.crs['init']
        self.image_proj = pyproj.Proj(init=image_crs)

    # TODO did i get the ordering right in the docs?
    def web_to_pixel(self, web_point):
        """Return point in pixel coordinates.

        Args:
            web_point: tuple (long, lat) in WebMercator coordinates

        Returns:
            tuple (row, col) in pixel coordinates
        """
        image_point = pyproj.transform(
            self.web_proj, self.image_proj, web_point[0], web_point[1])
        pixel_point = self.image_dataset.index(image_point[0], image_point[1])
        return pixel_point

    def pixel_to_web(self, pixel_point):
        """Return point in Web Mercator coordinates.

        Args:
            pixel_point: tuple (row, col) in pixel coordinates

        Returns:
            tuple (long, lat) in WebMercator coordinates
        """
        image_point = self.image_dataset.ul(
            int(pixel_point[0]), int(pixel_point[1]))
        web_point = pyproj.transform(
            self.image_proj, self.web_proj, image_point[0], image_point[1])
        return web_point
