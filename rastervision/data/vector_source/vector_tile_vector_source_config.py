from copy import deepcopy

import rastervision as rv
from rastervision.data.vector_source.vector_source_config import (
    VectorSourceConfig, VectorSourceConfigBuilder)
from rastervision.data.vector_source.vector_tile_vector_source import (
    VectorTileVectorSource)
from rastervision.data.vector_source.class_inference import ClassInferenceOptions


class VectorTileVectorSourceConfig(VectorSourceConfig):
    def __init__(self,
                 uri,
                 zoom,
                 id_field,
                 class_id_to_filter=None,
                 default_class_id=1,
                 line_bufs=None,
                 point_bufs=None):
        self.uri = uri
        self.zoom = zoom
        self.id_field = id_field
        super().__init__(
            rv.VECTOR_TILE_SOURCE,
            class_id_to_filter=class_id_to_filter,
            default_class_id=default_class_id,
            line_bufs=line_bufs,
            point_bufs=point_bufs)

    def to_proto(self):
        msg = super().to_proto()
        msg.mbtiles.uri = self.uri
        msg.mbtiles.zoom = self.zoom
        msg.mbtiles.id_field = self.id_field
        return msg

    def create_source(self, crs_transformer=None, extent=None, class_map=None):
        return VectorTileVectorSource(
            self.uri,
            self.zoom,
            self.id_field,
            crs_transformer,
            extent,
            line_bufs=self.line_bufs,
            point_bufs=self.point_bufs,
            class_inf_opts=ClassInferenceOptions(
                class_map=class_map,
                class_id_to_filter=self.class_id_to_filter,
                default_class_id=self.default_class_id))

    def report_io(self, command_type, io_def):
        # We shouldn't include self.uri as an input because it is just a URI schema
        # and the file checker will raise an error if it's included.
        pass


class VectorTileVectorSourceConfigBuilder(VectorSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'zoom': prev.zoom,
                'id_field': prev.id_field,
                'class_id_to_filter': prev.class_id_to_filter,
                'default_class_id': prev.default_class_id,
                'line_bufs': prev.line_bufs,
                'point_bufs': prev.point_bufs
            }

        super().__init__(VectorTileVectorSourceConfig, config)

    def validate(self):
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'VectorTileVectorSourceConfigBuilder requires uri which '
                'can be set using "with_uri".')

        if self.config.get('zoom') is None:
            raise rv.ConfigError(
                'VectorTileVectorSourceConfigBuilder requires zoom which '
                'can be set using "with_zoom".')

        # If not set explicitly, set it using default value.
        if self.config.get('id_field') is None:
            self.with_id_field()

        super().validate()

    def from_proto(self, msg):
        b = super().from_proto(msg)
        b = b.with_uri(msg.mbtiles.uri)
        b = b.with_zoom(msg.mbtiles.zoom)
        b = b.with_id_field(msg.mbtiles.id_field)
        return b

    def with_uri(self, uri):
        """Set the URI of the vector tiles.

        Args:
            uri: (str) URI of vector tile endpoint. Should either contain
                {z}/{x}/{y} or point to .mbtiles file.
        """
        b = deepcopy(self)
        b.config['uri'] = uri
        return b

    def with_zoom(self, zoom):
        """Set the zoom level to use when accessing vector tiles.

        Note: the vector tiles need to support the zoom level. Typically only
        a subset of zoom levels are supported.
        """
        b = deepcopy(self)
        b.config['zoom'] = zoom
        return b

    def with_id_field(self, id_field='@id'):
        """Set the name of the id field.

        Args:
            id_field: (str) name of field in feature['properties'] that
                contains the feature's unique id. Used for merging features
                that are split across tile boundaries.
        """
        b = deepcopy(self)
        b.config['id_field'] = id_field
        return b
