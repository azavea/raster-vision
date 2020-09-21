from pyproj import Transformer

from rasterio.transform import (rowcol, xy)

from rastervision.core.data.crs_transformer import (CRSTransformer,
                                                    IdentityCRSTransformer)


class RasterioCRSTransformer(CRSTransformer):
    """Transformer for a RasterioRasterSource."""

    def __init__(self, transform, image_crs, map_crs='epsg:4326'):
        """Constructor.

        Args:
            transform: Rasterio affine transform
            image_crs: CRS of image in format that PyProj can handle eg. wkt or init
                string
            map_crs: CRS of the labels
        """
        self.map2image = Transformer.from_crs(
            map_crs, image_crs, always_xy=True)
        self.image2map = Transformer.from_crs(
            image_crs, map_crs, always_xy=True)
        super().__init__(transform, image_crs, map_crs)

    def map_to_pixel(self, map_point):
        """Transform point from map to pixel-based coordinates.

        Args:
            map_point: (x, y) tuple in map coordinates

        Returns:
            (x, y) tuple in pixel coordinates
        """
        image_point = self.map2image.transform(*map_point)
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
        map_point = self.image2map.transform(*image_point)
        return map_point

    @classmethod
    def from_dataset(cls, dataset, map_crs='epsg:4326'):
        if dataset.crs is None:
            return IdentityCRSTransformer()
        transform = dataset.transform
        image_crs = dataset.crs.wkt
        return cls(transform, image_crs, map_crs)
