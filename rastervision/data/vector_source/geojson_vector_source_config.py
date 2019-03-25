from copy import deepcopy

import rastervision as rv
from rastervision.data.vector_source.vector_source_config import (
    VectorSourceConfig, VectorSourceConfigBuilder)
from rastervision.data.vector_source.geojson_vector_source import GeoJSONVectorSource
from rastervision.data.vector_source.class_inference import ClassInferenceOptions


class GeoJSONVectorSourceConfig(VectorSourceConfig):
    def __init__(self,
                 uri,
                 class_id_to_filter=None,
                 default_class_id=1,
                 line_bufs=None,
                 point_bufs=None):
        self.uri = uri
        super().__init__(
            rv.GEOJSON_SOURCE,
            class_id_to_filter=class_id_to_filter,
            default_class_id=default_class_id,
            line_bufs=line_bufs,
            point_bufs=point_bufs)

    def to_proto(self):
        msg = super().to_proto()
        msg.geojson.uri = self.uri
        return msg

    def create_source(self, crs_transformer=None, extent=None, class_map=None):
        return GeoJSONVectorSource(
            self.uri,
            crs_transformer,
            line_bufs=self.line_bufs,
            point_bufs=self.point_bufs,
            class_inf_opts=ClassInferenceOptions(
                class_map=class_map,
                class_id_to_filter=self.class_id_to_filter,
                default_class_id=self.default_class_id))

    def report_io(self, command_type, io_def):
        io_def.add_input(self.uri)


class GeoJSONVectorSourceConfigBuilder(VectorSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {
                'uri': prev.uri,
                'class_id_to_filter': prev.class_id_to_filter,
                'default_class_id': prev.default_class_id,
                'line_bufs': prev.line_bufs,
                'point_bufs': prev.point_bufs
            }

        super().__init__(GeoJSONVectorSourceConfig, config)

    def validate(self):
        if self.config.get('uri') is None:
            raise rv.ConfigError(
                'GeoJSONVectorSourceConfigBuilder requires uri which '
                'can be set using "with_uri".')

        super().validate()

    def from_proto(self, msg):
        b = super().from_proto(msg)
        b = b.with_uri(msg.geojson.uri)
        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        b.config['uri'] = uri
        return b
