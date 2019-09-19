from copy import deepcopy

import rastervision as rv
from rastervision.data.label_source import (LabelSourceConfig,
                                            LabelSourceConfigBuilder,
                                            ObjectDetectionLabelSource)
from rastervision.data.vector_source import VectorSourceConfig
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg


class ObjectDetectionLabelSourceConfig(LabelSourceConfig):
    def __init__(self, vector_source):
        super().__init__(source_type=rv.OBJECT_DETECTION)
        self.vector_source = vector_source

    def to_proto(self):
        msg = super().to_proto()
        opts = LabelSourceConfigMsg.ObjectDetectionLabelSource(
            vector_source=self.vector_source.to_proto())
        msg.object_detection_label_source.CopyFrom(opts)
        return msg

    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        vector_source = self.vector_source.create_source(
            crs_transformer=crs_transformer,
            extent=extent,
            class_map=task_config.class_map)
        return ObjectDetectionLabelSource(vector_source, crs_transformer,
                                          task_config.class_map, extent)

    def update_for_command(self, command_type, experiment_config,
                           context=None):
        self.vector_source.update_for_command(command_type, experiment_config,
                                              context)

    def report_io(self, command_type, io_def):
        self.vector_source.report_io(command_type, io_def)


class ObjectDetectionLabelSourceConfigBuilder(LabelSourceConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {'vector_source': prev.vector_source}

        super().__init__(ObjectDetectionLabelSourceConfig, config)

    def validate(self):
        super().validate()
        vector_source = self.config.get('vector_source')
        if vector_source is None:
            raise rv.ConfigError(
                'You must set the vector_source for ObjectDetectionLabelSourceConfig'
                ' Use "with_vector_source".')
        if not isinstance(vector_source, VectorSourceConfig):
            raise rv.ConfigError(
                'vector source must be a child of class VectorSourceConfig, got {}'
                .format(type(vector_source)))
        if vector_source.has_null_class_bufs():
            raise rv.ConfigError(
                'Setting buffer to None for a class in the vector_source is not allowed '
                'for ObjectDetectionLabelSourceConfig.')

    def from_proto(self, msg):
        b = ObjectDetectionLabelSourceConfigBuilder()

        # Added for backwards compatibility.
        if msg.HasField('object_detection_geojson_source'):
            vector_source = msg.object_detection_geojson_source.uri
        else:
            vector_source = rv.VectorSourceConfig.from_proto(
                msg.object_detection_label_source.vector_source)

        return b.with_vector_source(vector_source)

    def with_vector_source(self, vector_source):
        """Set the vector_source.

        Args:
            vector_source (str or VectorSource) if a string, assume it is
                a URI and use the default provider to construct a VectorSource.
        """
        if isinstance(vector_source, str):
            return self.with_uri(vector_source)

        b = deepcopy(self)
        if isinstance(vector_source, VectorSourceConfig):
            b.config['vector_source'] = vector_source
        else:
            raise rv.ConfigError(
                'vector_source must be of type str or VectorSource')

        return b

    def with_uri(self, uri):
        b = deepcopy(self)
        provider = rv._registry.get_vector_source_default_provider(uri)
        b.config['vector_source'] = provider.construct(uri)
        return b
