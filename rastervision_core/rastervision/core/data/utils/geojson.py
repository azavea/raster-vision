from typing import (TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List,
                    Optional, Union)

from shapely.geometry import shape, mapping
from shapely.ops import transform

from rastervision.core.data.utils.misc import listify_uris

if TYPE_CHECKING:
    from rastervision.core.data.crs_transformer import CRSTransformer
    from shapely.geometry.base import BaseGeometry

MULTI_GEOM_TYPES = {'MultiPolygon', 'MultiPoint', 'MultiLineString'}


def geometry_to_feature(mapping: dict, properties: dict = {}) -> dict:
    """Convert a serialized geometry to a serialized GeoJSON feature."""
    already_a_feature = mapping.get('type') == 'Feature'
    if already_a_feature:
        return mapping
    return {'type': 'Feature', 'geometry': mapping, 'properties': properties}


def geometries_to_geojson(geometries: Iterable[dict]) -> dict:
    """Convert serialized geometries to a serialized GeoJSON FeatureCollection.
    """
    features = [geometry_to_feature(g) for g in geometries]
    return features_to_geojson(features)


def features_to_geojson(features: List[dict]) -> dict:
    """Convert GeoJSON-like mapping of Features to a FeatureCollection."""
    return {'type': 'FeatureCollection', 'features': features}


def map_features(func: Callable, geojson: dict) -> dict:
    """Map GeoJSON features to new features. Returns a new GeoJSON dict."""
    features_in = geojson['features']
    features_out = list(map(func, features_in))
    return features_to_geojson(features_out)


def map_geoms(func: Callable, geojson: dict) -> dict:
    """Map GeoJSON features to new features by applying func to geometries.

    Returns a new GeoJSON dict.
    """

    def feat_func(feature_in: dict) -> dict:
        # to shapely geometry
        geom_in = shape(feature_in['geometry'])
        # apply func
        geom_out = func(geom_in, feature=feature_in)
        # back to dict
        geom_out = mapping(geom_out)
        # new feature with updated geometry
        feature_out = geometry_to_feature(geom_out,
                                          feature_in.get('properties', {}))
        return feature_out

    return map_features(feat_func, geojson)


def geojson_to_geoms(geojson: dict) -> Iterator['BaseGeometry']:
    """Return the shapely geometry for each feature in the GeoJSON."""
    geoms = (shape(f['geometry']) for f in geojson['features'])
    return geoms


def filter_features(func: Callable, geojson: dict) -> dict:
    """Filter GeoJSON features. Returns a new GeoJSON dict."""
    features_in = geojson['features']
    features_out = list(filter(func, features_in))
    return features_to_geojson(features_out)


def is_empty_feature(f: dict) -> bool:
    """Check if a GeoJSON Feature lacks geometry info.

    This was added to handle empty geoms which appear when using
    OSM vector tiles.

    Args:
        f (dict): A GeoJSON-like mapping of a Feature.

    Returns:
        bool: Whether the feature contains any geometry.
    """
    g: Optional[dict] = f.get('geometry')
    if not g:
        return True
    no_geometries = not g.get('geometries')
    no_coordinates = not g.get('coordinates')
    is_empty = no_geometries and no_coordinates
    return is_empty


def remove_empty_features(geojson: dict) -> dict:
    """Remove Features from a FeatureCollection that lack geometry info.

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.

    Returns:
        dict: Filtered FeatureCollection.
    """
    return filter_features(lambda f: not is_empty_feature(f), geojson)


def split_multi_geometries(geojson: dict) -> dict:
    """Break any Features with composite geometries into multiple Features.

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.

    Returns:
        dict: FeatureCollection without composite geometries.
    """

    def split_geom(geom: 'BaseGeometry') -> List['BaseGeometry']:
        # Split GeometryCollection into list of geoms.
        if geom.geom_type == 'GeometryCollection':
            geoms = list(geom)
        else:
            geoms = [geom]
        # Split any MultiX to list of X.
        new_geoms = []
        for g in geoms:
            if g.geom_type in MULTI_GEOM_TYPES:
                new_geoms.extend(list(g))
            else:
                new_geoms.append(g)
        return new_geoms

    new_features = []
    for f in geojson['features']:
        geom = shape(f['geometry'])
        split_geoms = split_geom(geom)
        for g in split_geoms:
            new_feature = geometry_to_feature(
                mapping(g), f.get('properties', {}))
            new_features.append(new_feature)
    return features_to_geojson(new_features)


def map_to_pixel_coords(geojson: dict,
                        crs_transformer: 'CRSTransformer') -> dict:
    """Convert a GeoJSON dict from map to pixel coordinates."""

    def map2pix(x, y, z=None):
        return crs_transformer.map_to_pixel((x, y))

    return map_geoms(lambda g, **kw: transform(map2pix, g), geojson)


def pixel_to_map_coords(geojson: dict,
                        crs_transformer: 'CRSTransformer') -> dict:
    """Convert a GeoJSON dict from pixel to map coordinates."""

    def pix2map(x, y, z=None):
        return crs_transformer.pixel_to_map((x, y))

    return map_geoms(lambda g, **kw: transform(pix2map, g), geojson)


def simplify_polygons(geojson: dict) -> dict:
    """Simplify polygon geometries by applying .buffer(0).

    For Polygon geomtries, .buffer(0) can do the following:
    1. *Sometimes* break up a polygon with "bowties" into multiple polygons.
        (See https://github.com/shapely/shapely/issues/599.)
    2. *Sometimes* "simplify" polygons. See shapely documentation for buffer().

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.

    Returns:
        dict: FeatureCollection with simplified geometries.
    """

    def buffer_polygon(geom: 'BaseGeometry',
                       feature: Optional[dict] = None) -> 'BaseGeometry':
        if geom.geom_type == 'Polygon':
            # the resultant value can be a Polygon or a MultiPolygon
            return geom.buffer(0)
        return geom

    geojson_buffered = map_geoms(buffer_polygon, geojson)
    geojson_cleaned = remove_empty_features(geojson_buffered)
    geojson_split = split_multi_geometries(geojson_cleaned)
    return geojson_split


def buffer_geoms(geojson: dict,
                 geom_type: str,
                 class_bufs: Dict[int, Optional[float]] = {},
                 default_buf: Optional[float] = 1) -> dict:
    """Buffer geometries.

    Geometries in features without a class_id property will be ignored.

    Args:
        geojson (dict): A GeoJSON-like mapping of a FeatureCollection.
        geom_type (str): Shapely geometry type to apply the buffering to. Other
            types of geometries will not be affected.
        class_bufs (Dict[int, Optional[float]]): Optional
            mapping from class ID to buffer distance (in pixel units) for
            geom_type geometries.

    Returns:
        dict: FeatureCollection with buffered geometries.
    """

    def buffer_geom(geom: 'BaseGeometry',
                    feature: Optional[dict] = None) -> 'BaseGeometry':
        if geom.geom_type != geom_type:
            return geom

        has_class_id = (('properties' in feature)
                        and ('class_id' in feature['properties']))
        if not has_class_id:
            return geom

        class_id = feature['properties']['class_id']
        buf = class_bufs.get(class_id, default_buf)
        # If buf for the class_id was explicitly set as None, don't buffer.
        if buf is not None:
            geom = geom.buffer(buf)
        return geom

    geojson_buffered = map_geoms(buffer_geom, geojson)
    return geojson_buffered


def all_geoms_valid(geojson: dict):
    """Check if there are any invalid geometries in the GeoJSON."""
    geoms = geojson_to_geoms(geojson)
    return all(g.is_valid for g in geoms)


def get_polygons_from_uris(
        uris: Union[str, List[str]],
        crs_transformer: 'CRSTransformer') -> List['BaseGeometry']:
    """Load and return polygons (in pixel coords) from one or more URIs."""

    # use local imports to avoid circular import problems
    from rastervision.core.data import GeoJSONVectorSource

    polygons = []
    uris = listify_uris(uris)
    for uri in uris:
        source = GeoJSONVectorSource(
            uri=uri, ignore_crs_field=True, crs_transformer=crs_transformer)
        polygons += source.get_geoms()
    return polygons
