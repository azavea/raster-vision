from typing import TYPE_CHECKING, List
from abc import ABC, abstractmethod
import logging

from shapely.ops import unary_union
import geopandas as gpd

from rastervision.core.box import Box
from rastervision.core.data.utils import (
    remove_empty_features, split_multi_geometries, map_to_pixel_coords,
    pixel_to_map_coords, simplify_polygons, all_geoms_valid, geojson_to_geoms)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry
    from rastervision.core.data import CRSTransformer, VectorTransformer

log = logging.getLogger(__name__)


class VectorSource(ABC):
    """A source of vector data."""

    def __init__(self,
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = []):
        """Constructor.

        Args:
            crs_transformer: A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers: ``VectorTransformers`` for transforming
                geometries. Defaults to ``[]``.
        """
        self.crs_transformer = crs_transformer
        self.vector_transformers = vector_transformers
        self._geojson = None
        self._extent = None

    def get_geojson(self, to_map_coords: bool = False) -> dict:
        """Return transformed GeoJSON.

        This makes the following transformations to the raw geojson:

        - converts to pixels coords (by default)
        - removes empty features
        - splits apart multi-geoms and geom collections into single geometries
        - buffers lines and points into Polygons

        Additionally, the transformations specified by all the
        VectorTransformers in vector_transformers are also applied.

        Args:
            to_map_coords: If true, will return GeoJSON in map coordinates.

        Returns:
            dict in GeoJSON format
        """
        if self._geojson is not None:
            return self._geojson

        geojson = self._get_geojson()
        geojson = sanitize_geojson(
            geojson, self.crs_transformer, to_map_coords=to_map_coords)

        for tf in self.vector_transformers:
            geojson = tf(geojson, crs_transformer=self.crs_transformer)

        self._geojson = geojson
        return geojson

    def get_geoms(self, to_map_coords: bool = False) -> List['BaseGeometry']:
        """Returns all geometries in the transformed GeoJSON as Shapely geoms.

        Args:
            to_map_coords: If true, will return geoms in map coordinates.

        Returns:
            List['BaseGeometry']: List of Shapely geoms.
        """
        geojson = self.get_geojson(to_map_coords=to_map_coords)
        return list(geojson_to_geoms(geojson))

    @abstractmethod
    def _get_geojson(self) -> dict:
        """Return raw GeoJSON."""
        pass

    def get_dataframe(self, to_map_coords: bool = False) -> gpd.GeoDataFrame:
        """Return geometries as a :class:`~geopandas.GeoDataFrame`."""
        geojson = self.get_geojson(to_map_coords=to_map_coords)
        df = gpd.GeoDataFrame.from_features(geojson)
        if len(df) == 0 and 'geometry' not in df.columns:
            df.loc[:, 'geometry'] = []
        return df

    @property
    def extent(self) -> Box:
        """Envelope of union of all geoms."""
        if self._extent is None:
            envelope = unary_union(self.get_geoms()).envelope
            self._extent = Box.from_shapely(envelope).to_int()
        return self._extent


def sanitize_geojson(geojson: dict,
                     crs_transformer: 'CRSTransformer',
                     to_map_coords: bool = False) -> dict:
    """Apply some basic transformations (listed below) to a GeoJSON.

    The following transformations are applied:

    1.  Removal of features without geometries.
    2.  Coordinate transformation to pixel coordinates.
    3.  Splitting of multi-part geometries e.g. MultiPolygon --> Polygons.
    4.  (Optional) If to_map_coords=true, transformation back to map
        coordinates.

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.
        crs_transformer (CRSTransformer): A CRS transformer for coordinate
            transformation.
        to_map_coords (bool, optional): If True, transform geometries back to
            map coordinates before returning. Defaults to False.

    Returns:
        dict: Transformed FeatureCollection.
    """
    geojson = remove_empty_features(geojson)
    geojson = map_to_pixel_coords(geojson, crs_transformer)
    geojson = split_multi_geometries(geojson)
    geojson = simplify_polygons(geojson)
    if to_map_coords:
        geojson = pixel_to_map_coords(geojson, crs_transformer)
    if not all_geoms_valid(geojson):
        log.warn(f'Invalid geometries found in features in the GeoJSON.')
    return geojson
