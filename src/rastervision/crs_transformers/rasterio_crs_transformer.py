import pyproj

from rastervision.core.crs_transformer import CRSTransformer


class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource.

    This assumes that the map coordinates are always in lon/lat format.
    """

    def __init__(self, image_dataset):
        """Construct transformer.

        Args:
            image_dataset: Rasterio DatasetReader
        """
        self.image_dataset = image_dataset
        self.map_proj = pyproj.Proj(init='epsg:4326')
        image_crs = image_dataset.crs['init']
        self.image_proj = pyproj.Proj(init=image_crs)

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (long, lat) tuple

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = pyproj.transform(
            self.map_proj, self.image_proj, map_point[0], map_point[1])
        pixel_point = self.image_dataset.index(image_point[0], image_point[1])
        pixel_point = (pixel_point[1], pixel_point[0])
        return pixel_point

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (lon, lat) tuple
        """
        image_point = self.image_dataset.ul(
            int(pixel_point[1]), int(pixel_point[0]))
        map_point = pyproj.transform(
            self.image_proj, self.map_proj, image_point[0], image_point[1])
        return map_point
