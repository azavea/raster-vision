from typing import TYPE_CHECKING, Optional
import numpy as np

from shapely.ops import transform

from rastervision.core.data.crs_transformer import RasterioCRSTransformer
from rastervision.core.data.utils.geojson import (
    pixel_to_map_coords, map_to_pixel_coords, map_geoms)
from rastervision.core.data.vector_transformer import VectorTransformer

if TYPE_CHECKING:
    from rastervision.core.data import CRSTransformer

METERS_PER_DEGREE = 111319.5
RADIANS_PER_DEGREE = np.pi / 180


class ShiftTransformer(VectorTransformer):
    """Shift geometries by some distance specified in meters."""

    def __init__(self,
                 x_shift: float = 0.,
                 y_shift: float = 0.,
                 round_pixels: bool = True):
        """Constructor.

        Args:
        """
        self.x_shift = x_shift
        self.y_shift = y_shift
        self.round_pixels = round_pixels

    def transform(self,
                  geojson: dict,
                  crs_transformer: Optional['CRSTransformer'] = None) -> dict:

        # https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters  # noqa
        def shift(x, y, z=None):
            lon, lat = x, y
            lat_radians = RADIANS_PER_DEGREE * lat
            dlon = self.x_shift / (METERS_PER_DEGREE * np.cos(lat_radians))
            dlat = self.y_shift / METERS_PER_DEGREE
            return lon + dlon, lat + dlat

        wgs84_transformer = self.make_wgs84_transformer(crs_transformer)

        geojson_pixel = geojson
        geojson_wgs84 = pixel_to_map_coords(geojson_pixel, wgs84_transformer)
        geojson_wgs84_shifted = map_geoms(lambda g, **kw: transform(shift, g),
                                          geojson_wgs84)
        geojson_pixel_shifted = map_to_pixel_coords(geojson_wgs84_shifted,
                                                    wgs84_transformer)

        return geojson_pixel_shifted

    def make_wgs84_transformer(self, crs_transformer: 'CRSTransformer'):
        wgs84_transformer = RasterioCRSTransformer(
            transform=crs_transformer.transform,
            image_crs=crs_transformer.image_crs,
            map_crs='epsg:4326',
            round_pixels=self.round_pixels)
        return wgs84_transformer
