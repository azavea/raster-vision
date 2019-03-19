from abc import ABC, abstractmethod

from shapely.geometry import shape, mapping
import shapely

from rastervision.data.vector_source.class_inference import (
    ClassInference, ClassInferenceOptions)


def transform_geojson(geojson,
                      crs_transformer,
                      line_bufs=None,
                      point_bufs=None,
                      to_map_coords=False):
    def is_empty_feat(f):
        # This was added to handle empty geoms which appear when using
        # OSM vector tiles.
        return ((not f.get('geometry'))
                or ((not f['geometry'].get('coordinates')) and
                    (not f['geometry'].get('geometries'))))

    new_features = []
    for f in geojson['features']:
        if is_empty_feat(f):
            continue

        geom = shape(f['geometry'])

        # Convert map to pixel coords. We need to convert to pixel coords before applying
        # buffering because those values are assumed to be in pixel units.
        def m2p(x, y, z=None):
            return crs_transformer.map_to_pixel((x, y))

        geom = shapely.ops.transform(m2p, geom)

        # Split GeometryCollection into list of geoms.
        geoms = [geom]
        if geom.geom_type == 'GeometryCollection':
            geoms = list(geom)

        # Split any MultiX to list of X.
        new_geoms = []
        for g in geoms:
            if g.geom_type in [
                    'MultiPolygon', 'MultiPoint', 'MultiLineString'
            ]:
                new_geoms.extend(list(g))
            else:
                new_geoms.append(g)
        geoms = new_geoms

        # Buffer geoms.
        class_id = f['properties']['class_id']
        new_geoms = []
        for g in geoms:
            if g.geom_type == 'LineString':
                line_buf = 1
                if line_bufs is not None:
                    line_buf = line_bufs.get(class_id, 1)
                # If line_buf for the class_id was explicitly set as None, then
                # don't buffer.
                if line_buf is not None:
                    g = g.buffer(line_buf)
                new_geoms.append(g)
            elif g.geom_type == 'Point':
                point_buf = 1
                if point_bufs is not None:
                    point_buf = point_bufs.get(class_id, 1)
                # If point_buf for the class_id was explicitly set as None, then
                # don't buffer.
                if point_buf is not None:
                    g = g.buffer(point_buf)
                new_geoms.append(g)
            else:
                # Use buffer trick to handle self-intersecting polygons. Buffer returns
                # a MultiPolygon if there is a bowtie, so we have to convert it to a
                # list of Polygons.
                poly_buf = g.buffer(0)
                if poly_buf.geom_type == 'MultiPolygon':
                    new_geoms.extend(list(poly_buf))
                else:
                    new_geoms.append(poly_buf)
        geoms = new_geoms

        # Convert back to map coords if desired. This is here so the QGIS plugin can
        # take the GeoJSON produced by a VectorSource and display it on a map.
        if to_map_coords:

            def p2m(x, y, z=None):
                return crs_transformer.pixel_to_map((x, y))

            geoms = [shapely.ops.transform(p2m, g) for g in geoms]

        for g in geoms:
            new_f = {
                'type': 'Feature',
                'geometry': mapping(g),
                'properties': f['properties']
            }
            # Have to check for empty features again which could have been introduced
            # when splitting apart multi-geoms.
            if not is_empty_feat(new_f):
                new_features.append(new_f)

    return {'type': 'FeatureCollection', 'features': new_features}


class VectorSource(ABC):
    """A source of vector data."""

    def __init__(self,
                 crs_transformer,
                 line_bufs=None,
                 point_bufs=None,
                 class_inf_opts=None):
        """Constructor.

        Args:
            crs_transformer: (CRSTransformer)
            line_bufs: (dict or None) If none, uses default buffer value of 1. Otherwise,
                a map from class_id to number of pixels to buffer by. If the buffer value
                is None, then no buffering will be performed and the LineString or Point
                won't get converted to a Polygon. Not converting to Polygon is
                incompatible with the currently available LabelSources, but may be useful
                in the future.
            point_bufs: (dict or None) same as above, but used for buffering Points into
                Polygons.
            class_inf_opts: (ClassInferenceOptions)
        """
        self.crs_transformer = crs_transformer
        self.line_bufs = line_bufs
        self.point_bufs = point_bufs
        if class_inf_opts is None:
            class_inf_opts = ClassInferenceOptions()
        self.class_inference = ClassInference(class_inf_opts)

        self.geojson = None

    def get_geojson(self, to_map_coords=False):
        """Return normalized GeoJSON.

        This infers a class_id property for each feature, converts to pixels coords (by
        default), removes empty features, splits apart multi-geoms and geom collections
        into single geometries, and buffers lines and points into Polygons.

        Args:
            to_map_coords: If true, will return GeoJSON in map coordinates.

        Returns:
            dict in GeoJSON format
        """
        if self.geojson is None:
            self.geojson = self._get_geojson()

        geojson = transform_geojson(
            self.geojson,
            self.crs_transformer,
            self.line_bufs,
            self.point_bufs,
            to_map_coords=to_map_coords)

        return geojson

    @abstractmethod
    def _get_geojson(self):
        """Return GeoJSON with class_ids in the properties."""
        pass
