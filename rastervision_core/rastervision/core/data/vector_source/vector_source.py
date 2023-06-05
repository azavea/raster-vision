from typing import TYPE_CHECKING, List, Optional
from abc import ABC, abstractmethod
import logging

import geopandas as gpd

from rastervision.core.box import Box
from rastervision.core.data.utils import (
    remove_empty_features, split_multi_geometries, map_to_pixel_coords,
    pixel_to_map_coords, simplify_polygons, all_geoms_valid, geojson_to_geoms,
    geojson_to_geodataframe, get_geojson_extent, filter_geojson_to_window)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry
    from rastervision.core.data import CRSTransformer, VectorTransformer

log = logging.getLogger(__name__)


class VectorSource(ABC):
    """A source of vector data."""

    def __init__(self,
                 crs_transformer: 'CRSTransformer',
                 vector_transformers: List['VectorTransformer'] = [],
                 bbox: Optional[Box] = None):
        """Constructor.

        Args:
            crs_transformer (CRSTransformer): A ``CRSTransformer`` to convert
                between map and pixel coords. Normally this is obtained from a
                :class:`.RasterSource`.
            vector_transformers (List[VectorTransformer]):
                ``VectorTransformers`` for transforming geometries.
                Defaults to ``[]``.
            bbox (Optional[Box]): User-specified crop of the extent. If None,
                the full extent available in the source file is used.
        """
        self.crs_transformer = crs_transformer
        self.vector_transformers = vector_transformers
        self._geojson = None
        self._gdf = None
        self._extent = None
        self._bbox = bbox

    def get_geojson(self,
                    window: Optional[Box] = None,
                    to_map_coords: bool = False) -> dict:
        """Return transformed GeoJSON.

        This makes the following transformations to the raw geojson:

        - converts to pixels coords (by default)
        - removes empty features
        - splits apart multi-geoms and geom collections into single geometries
        - buffers lines and points into Polygons

        Additionally, the transformations specified by all the
        VectorTransformers in vector_transformers are also applied.

        Args:
            window (Optional[Box]): If specified, return only the features that
                intersect with this window; otherwise, return all features.
                Defaults to None.
            to_map_coords (bool): If true, will return GeoJSON in map
                coordinates.

        Returns:
            dict in GeoJSON format
        """
        if self._geojson is not None:
            return self._geojson

        geojson = self._get_geojson()
        geojson = sanitize_geojson(
            geojson, self.crs_transformer, to_map_coords=to_map_coords)

        if self._bbox is not None:
            geojson = filter_geojson_to_window(geojson, self.bbox)

        for tf in self.vector_transformers:
            geojson = tf(geojson, crs_transformer=self.crs_transformer)

        self._geojson = geojson

        if window is not None:
            return filter_geojson_to_window(geojson, window)
        return geojson

    def get_geoms(self,
                  window: Optional[Box] = None,
                  to_map_coords: bool = False) -> List['BaseGeometry']:
        """Returns all geometries in the transformed GeoJSON as Shapely geoms.

        Args:
            to_map_coords: If true, will return geoms in map coordinates.

        Returns:
            List['BaseGeometry']: List of Shapely geoms.
        """
        geojson = self.get_geojson(window=window, to_map_coords=to_map_coords)
        return list(geojson_to_geoms(geojson))

    @abstractmethod
    def _get_geojson(self) -> dict:
        """Return raw GeoJSON."""
        pass

    def get_dataframe(self,
                      window: Optional[Box] = None,
                      to_map_coords: bool = False) -> gpd.GeoDataFrame:
        """Return geometries as a :class:`~geopandas.GeoDataFrame`.

        Arguments:
            window (Optional[Box]): If specified, return only the features that
                intersect with this window; otherwise, return all features.
                Defaults to None.
            to_map_coords (bool): If true, will return GeoJSON in map
                coordinates.
        """
        if self._gdf is None:
            geojson = self.get_geojson(to_map_coords=to_map_coords)
            self._gdf = geojson_to_geodataframe(geojson)
        gdf = self._gdf
        if window is not None:
            window_geom = window.to_shapely()
            gdf: gpd.GeoDataFrame = gdf[gdf.intersects(window_geom)]
        return gdf

    @property
    def extent(self) -> Box:
        """Envelope of union of all geoms."""
        if self._extent is None:
            self._extent = get_geojson_extent(self.get_geojson())
        return self._extent

    @property
    def bbox(self) -> 'Box':
        """Bounding box applied to the source data."""
        if self._bbox is None:
            self._bbox = self.extent
        return self._bbox


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
