from copy import deepcopy

import rastervision as rv
from rastervision.data.vector_source.vector_source_config import (
    VectorSourceConfig, VectorSourceConfigBuilder)
from rastervision.data.vector_source.mbtiles_vector_source import MBTilesVectorSource
from rastervision.data.vector_source.class_inference import ClassInferenceOptions


class MBTilesVectorSourceConfig(VectorSourceConfig):
    def __init__(self, uri, zoom, class_id_to_filter=None, default_class_id=1):
        self.uri = uri
        self.zoom = zoom
        super().__init__(
            rv.MBTILES_SOURCE,
            class_id_to_filter=class_id_to_filter,
            default_class_id=default_class_id)

    def to_proto(self):
        msg = super().to_proto()
        msg.mbtiles.uri = self.uri
        msg.mbtiles.zoom = self.zoom
        return msg

    def create_source(self, crs_transformer=None, extent=None, class_map=None):
        return MBTilesVectorSource(
            self.uri,
            self.zoom,
            crs_transformer,
            extent,
            class_inf_opts=ClassInferenceOptions(
                class_map=class_map,
                class_id_to_filter=self.class_id_to_filter,
                default_class_id=self.default_class_id))

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        # We shouldn't include self.uri as an input because it is just a URI schema
        # and the file checker will raise an error if it's included.
        pass


class MBTilesVectorSourceConfigBuilder(VectorSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'zoom': prev.zoom,
                'class_id_to_filter': prev.class_id_to_filter,
                'default_class_id': prev.default_class_id
            }

        super().__init__(MBTilesVectorSourceConfig, config)

    def validate(self):
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'MBTilesVectorSourceConfigBuilder requires uri which '
                'can be set using "with_uri".')

        if self.config.get('zoom') is None:
            raise rv.ConfigError(
                'MBTilesVectorSourceConfigBuilder requires zoom which '
                'can be set using "with_zoom".')

        super().validate()

    def from_proto(self, msg):
        b = super().from_proto(msg)
        b = b.with_uri(msg.mbtiles.uri)
        b = b.with_zoom(msg.mbtiles.zoom)
        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        b.config['uri'] = uri
        return b

    def with_zoom(self, zoom):
        """Set the zoom level to use when fetching vector tiles.

        Note: the vector tiles endpoint needs to support the zoom level. Typically only
        a subset of zoom levels are supported.
        """
        b = deepcopy(self)
        b.config['zoom'] = zoom
        return b
