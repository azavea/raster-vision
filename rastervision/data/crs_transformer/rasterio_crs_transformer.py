import pyproj

from rasterio.transform import (rowcol, xy)

from rastervision.data.crs_transformer import (CRSTransformer,
                                               IdentityCRSTransformer)


class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource."""

    def __init__(self, transform, image_crs, map_crs='epsg:4326'):
        """Construct transformer.

        Args:
            image_dataset: Rasterio DatasetReader
            map_crs: CRS code
        """
        self.map_proj = pyproj.Proj(init=map_crs)
        self.image_proj = pyproj.Proj(image_crs)

        super().__init__(image_crs, map_crs, transform)

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = pyproj.transform(self.map_proj, self.image_proj,
                                       map_point[0], map_point[1])
        pixel_point = rowcol(self.transform, image_point[0], image_point[1])
        pixel_point = (pixel_point[1], pixel_point[0])
        return pixel_point

    def pixel_to_map(self, pixel_point):
        """Transform point from pixel to map-based coordinates.

        Args:
            pixel_point: (x, y) tuple in pixel coordinates

        Returns:
            (x, y) tuple in map coordinates
        """
        image_point = xy(self.transform, int(pixel_point[1]),
                         int(pixel_point[0]))
        map_point = pyproj.transform(self.image_proj, self.map_proj,
                                     image_point[0], image_point[1])
        return map_point

    @classmethod
    def from_dataset(cls, dataset, map_crs='epsg:4326'):
        if dataset.crs is None:
            return IdentityCRSTransformer()
        transform = dataset.transform
        image_crs = dataset.crs
        return cls(transform, image_crs, map_crs)

    def get_affine_transform(self):
        return self.transform
