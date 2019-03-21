import json
import logging
import copy
from subprocess import check_output
import os
from functools import lru_cache

from supermercado.burntiles import burn
from shapely.geometry import shape, mapping
from shapely.ops import cascaded_union

from rastervision.data.vector_source.vector_source import VectorSource
from rastervision.utils.files import get_cached_file
from rastervision.rv_config import RVConfig

log = logging.getLogger(__name__)


def merge_geojson(geojson, id_field):
    """Merge features that share an id.

    Args:
        geojson: (dict) GeoJSON with features that have an id property used to merge
            features that are split across multiple tiles.
        id_field: (str) name of field in feature['properties'] that contains the
            feature's unique id. Used for merging features that are split across
            tile boundaries.
    """
    id_to_features = {}
    for f in geojson['features']:
        id = f['properties'][id_field]
        id_features = id_to_features.get(id, [])
        id_features.append(f)
        id_to_features[id] = id_features

    proc_features = []
    for id, id_features in id_to_features.items():
        id_geoms = []
        for f in id_features:
            g = shape(f['geometry'])
            # Self-intersection trick.
            if f['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                g = g.buffer(0)
            id_geoms.append(g)

        union_geom = cascaded_union(id_geoms)
        union_feature = copy.deepcopy(id_features[0])
        union_feature['geometry'] = mapping(union_geom)
        proc_features.append(union_feature)

    return {'type': 'FeatureCollection', 'features': proc_features}


@lru_cache(maxsize=32)
def get_tile_features(tile_uri, z, x, y):
    """Get GeoJSON features for a specific tile using in-memory caching."""
    cache_dir = os.path.join(RVConfig.get_tmp_dir_root(), 'vector-tiles')
    tile_path = get_cached_file(cache_dir, tile_uri)
    cmd = ['tippecanoe-decode', '-f', '-c', tile_path, str(z), str(x), str(y)]
    tile_geojson_str = check_output(cmd).decode('utf-8')
    tile_features = [json.loads(ts) for ts in tile_geojson_str.split('\n')]

    return tile_features


def vector_tile_to_geojson(uri, zoom, map_extent):
    """Get GeoJSON features that overlap with an extent from a vector tile endpoint."""
    log.info('Downloading and converting vector tiles to GeoJSON...')

    # Figure out which tiles cover the extent.
    extent_polys = [{
        'type': 'Feature',
        'properties': {},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [map_extent.geojson_coordinates()]
        }
    }]
    xyzs = burn(extent_polys, zoom)

    # Retrieve tile features.
    features = []
    for xyz in xyzs:
        x, y, z = xyz
        # If this isn't a zxy schema, this is a no-op.
        tile_uri = uri.format(x=x, y=y, z=z)
        tile_features = get_tile_features(tile_uri, z, x, y)
        features.extend(tile_features)

    # Crop features to extent
    extent_geom = map_extent.to_shapely()
    cropped_features = []
    for f in features:
        geom = shape(f['geometry'])
        if f['geometry']['type'].lower() in ['polygon', 'multipolygon']:
            geom = geom.buffer(0)
        geom = geom.intersection(extent_geom)
        if not geom.is_empty:
            f = dict(f)
            f['geometry'] = mapping(geom)
            cropped_features.append(f)

    return {'type': 'FeatureCollection', 'features': cropped_features}


class VectorTileVectorSource(VectorSource):
    def __init__(self,
                 uri,
                 zoom,
                 id_field,
                 crs_transformer,
                 extent,
                 line_bufs=None,
                 point_bufs=None,
                 class_inf_opts=None):
        """Constructor.

        Args:
            uri: (str) URI of vector tile endpoint. Should either contain {z}/{x}/{y} or
                point to .mbtiles file.
            zoom: (int) valid zoom level to use when fetching tiles from endpoint
            id_field: (str) name of field in feature['properties'] that contains the
                feature's unique id. Used for merging features that are split across
                tile boundaries.
            crs_transformer: (CRSTransformer)
            extent: (Box) extent of scene which determines which features to return
            class_inf_opts: (ClassInferenceOptions)
        """
        self.uri = uri
        self.zoom = zoom
        self.id_field = id_field
        self.extent = extent
        super().__init__(crs_transformer, line_bufs, point_bufs,
                         class_inf_opts)

    def _get_geojson(self):
        # This attempts to do things in an efficient order. First, we extract raw
        # GeoJSON from the vector tiles covering the extent. This uses caching, so that
        # we never have to process the same vector tile twice (even across scenes). Then,
        # we infer class ids, which drops any irrelevant features which should speed up
        # the next phase which merges features that are split across tiles.
        map_extent = self.extent.reproject(
            lambda point: self.crs_transformer.pixel_to_map(point))
        log.debug(
            'Reading and converting vector tiles to GeoJSON for extent...')
        geojson = vector_tile_to_geojson(self.uri, self.zoom, map_extent)
        log.debug('Inferring class_ids...')
        geojson = self.class_inference.transform_geojson(geojson)
        log.debug('Merging GeoJSON features...')
        geojson = merge_geojson(geojson, self.id_field)
        return geojson
