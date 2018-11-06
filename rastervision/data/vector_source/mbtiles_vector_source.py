import json
import logging
import copy
from subprocess import check_output

from supermercado.burntiles import burn
from shapely.geometry import shape, mapping
from shapely.ops import cascaded_union

from rastervision.data.vector_source.vector_source import VectorSource
from rastervision.utils.files import download_if_needed
from rastervision.rv_config import RVConfig

log = logging.getLogger(__name__)


def process_features(features, map_extent):
    """Merge features that share an id and crop them against the extent.

    Args:
        features: (list) GeoJSON feature that have a __id property used to merge
            features that are split across multiple tiles.
        map_extent: (Box) in map coordinates
    """
    extent_geom = map_extent.to_shapely()
    id_to_features = {}
    for f in features:
        id = f['properties']['__id']
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
        geom = union_geom.intersection(extent_geom)
        union_feature = copy.deepcopy(id_features[0])
        union_feature['geometry'] = mapping(geom)
        proc_features.append(union_feature)
    return proc_features


def mbtiles_to_geojson(uri, zoom, crs_transformer, extent):
    """Get GeoJSON covering an extent from a vector tile endpoint.

    Merges features that are split across tiles and crops against the extentself.
    """
    log.info('Downloading and converting vector tiles to GeoJSON...')

    # Figure out which tiles cover the extent.
    map_extent = extent.reproject(
        lambda point: crs_transformer.pixel_to_map(point))
    extent_polys = [{
        'type': 'Feature',
        'properties': {},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [map_extent.geojson_coordinates()]
        }
    }]
    xyzs = burn(extent_polys, zoom)

    # Download tiles and convert to geojson.
    features = []
    for xyz in xyzs:
        x, y, z = xyz
        tile_uri = uri.format(x=x, y=y, z=z)

        with RVConfig.get_tmp_dir() as tmp_dir:
            # TODO some opportunities for improving efficiency:
            # * LRU in memory cache
            # * Filter out features that have None as class_id before calling
            # process_features
            tile_path = download_if_needed(tile_uri, tmp_dir)
            cmd = [
                'tippecanoe-decode', '-f', '-c', tile_path,
                str(z),
                str(x),
                str(y)
            ]
            tile_geojson_str = check_output(cmd).decode('utf-8')
            tile_features = [
                json.loads(ts) for ts in tile_geojson_str.split('\n')
            ]
            features.extend(tile_features)

    proc_features = process_features(features, map_extent)
    geojson = {'type': 'FeatureCollection', 'features': proc_features}
    return geojson


class MBTilesVectorSource(VectorSource):
    def __init__(self, uri, zoom, crs_transformer, extent,
                 class_inf_opts=None):
        """Constructor.

        Args:
            uri: (str) URI of Mapbox vector tiles endpoint. Should contain {z}/{x}/{y}
            zoom: (int) valid zoom level to use when fetching tiles from endpoint
            crs_transformer: (CRSTransformer)
            extent: (Box) extent of scene which determines which features to return
            class_inf_opts: (ClassInferenceOptions)
        """
        self.uri = uri
        self.zoom = zoom
        self.crs_transformer = crs_transformer
        self.extent = extent
        super().__init__(class_inf_opts)

    def _get_geojson(self):
        return mbtiles_to_geojson(self.uri, self.zoom, self.crs_transformer,
                                  self.extent)
