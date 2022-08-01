from typing import TYPE_CHECKING, Dict, Optional
from abc import ABC, abstractmethod
import logging

from rastervision.core.data.utils import (
    remove_empty_features, split_multi_geometries, map_to_pixel_coords,
    pixel_to_map_coords, buffer_geoms, simplify_polygons, all_geoms_valid)
from rastervision.core.data.vector_source.class_inference import (
    ClassInference)

if TYPE_CHECKING:
    from rastervision.core.data.vector_source.vector_source_config import (
        VectorSourceConfig)
    from rastervision.core.data.class_config import ClassConfig
    from rastervision.core.data.crs_transformer import CRSTransformer

log = logging.getLogger(__name__)


class VectorSource(ABC):
    """A source of vector data."""

    def __init__(self, vs_config: 'VectorSourceConfig',
                 class_config: 'ClassConfig',
                 crs_transformer: 'CRSTransformer'):
        self.vs_config = vs_config
        self.class_config = class_config
        self.crs_transformer = crs_transformer
        self.class_inference = ClassInference(
            vs_config.default_class_id,
            class_config=class_config,
            class_id_to_filter=vs_config.class_id_to_filter)

    def get_geojson(self, to_map_coords=False):
        """Return normalized GeoJSON.

        This makes the following transformations to the raw geojson:
        - infers a class_id property for each feature
        - converts to pixels coords (by default)
        - removes empty features
        - splits apart multi-geoms and geom collections into single geometries
        - buffers lines and points into Polygons

        Args:
            to_map_coords: If true, will return GeoJSON in map coordinates.

        Returns:
            dict in GeoJSON format
        """
        geojson_with_class_ids = self._get_geojson()
        geojson_transformed = transform_geojson(
            geojson_with_class_ids,
            self.crs_transformer,
            self.vs_config.line_bufs,
            self.vs_config.point_bufs,
            to_map_coords=to_map_coords)

        return geojson_transformed

    @abstractmethod
    def _get_geojson(self) -> dict:
        """Return GeoJSON with class_ids in the properties."""
        pass


def transform_geojson(geojson: dict,
                      crs_transformer: 'CRSTransformer',
                      line_bufs: Optional[Dict[int, Optional[float]]] = None,
                      point_bufs: Optional[Dict[int, Optional[float]]] = None,
                      to_map_coords: bool = False) -> dict:
    """Apply some transformations (listed below) to a FeatureCollection.

    The following transformations are applied:
    1. Removal of features without geometries.
    2. Coordinate transformation to pixel coordinates.
    3. Splitting of composite geometries e.g. MultiPolygon --> Polygons.
    4. Buffering of geometries.
    5. (Optional) If to_map_coords=true, transformation back to map
    coordinates.

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.
        crs_transformer (CRSTransformer): A CRS transformer for coordinate
            transformation.
        line_bufs (Optional[Dict[int, Optional[float]]], optional): Optional
            mapping from class ID to buffer distance (in pixel units) for
            LineString geometries. If None, a buffering of 1 unit is applied.
        point_bufs (Optional[Dict[int, Optional[float]]], optional): Optional
            mapping from class ID to buffer distance (in pixel units) for
            Point geometries. If None, a buffering of 1 unit is applied.
        to_map_coords (bool, optional): If True, transform geometries back to
            map coordinates before returning. Defaults to False.

    Returns:
        dict: Transformed FeatureCollection.
    """
    if point_bufs is None:
        point_bufs = {}
    if line_bufs is None:
        line_bufs = {}
    geojson = remove_empty_features(geojson)
    geojson = map_to_pixel_coords(geojson, crs_transformer)
    geojson = split_multi_geometries(geojson)
    geojson = buffer_geoms(geojson, geom_type='Point', class_bufs=point_bufs)
    geojson = buffer_geoms(
        geojson, geom_type='LineString', class_bufs=line_bufs)
    geojson = simplify_polygons(geojson)
    if to_map_coords:
        geojson = pixel_to_map_coords(geojson, crs_transformer)
    if not all_geoms_valid(geojson):
        log.warn(f'Invalid geometries found in features in the GeoJSON.')
    return geojson
