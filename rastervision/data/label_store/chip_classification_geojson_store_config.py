import os
from copy import deepcopy

import rastervision as rv
from rastervision.data.label_store import (
    LabelStoreConfig, LabelStoreConfigBuilder, ChipClassificationGeoJSONStore)


class ChipClassificationGeoJSONStoreConfig(LabelStoreConfig):
    def __init__(self, uri=None):
        super().__init__(store_type=rv.CHIP_CLASSIFICATION_GEOJSON)
        self.uri = uri

    def to_proto(self):
        msg = super().to_proto()
        if self.uri:
            msg.uri = self.uri
        return msg

    def for_prediction(self, label_uri):
        return self.to_builder() \
                   .with_uri(label_uri) \
                   .build()

    def create_store(self, task_config, extent, crs_transformer, tmp_dir):
        return ChipClassificationGeoJSONStore(self.uri, crs_transformer,
                                              task_config.class_map)

    def update_for_command(self, command_type, experiment_config, context=[]):
        conf = self
        io_def = rv.core.CommandIODefinition()

        if command_type == rv.PREDICT:
            if not self.uri:
                # Construct the  URI for this prediction store,
                # using the scene ID.
                root = experiment_config.predict_uri
                uri = None
                for c in context:
                    if isinstance(c, rv.SceneConfig):
                        uri = os.path.join(root, '{}.json'.format(c.id))
                if uri:
                    conf = conf.to_builder() \
                               .with_uri(uri) \
                               .build()
                    io_def.add_output(uri)
                else:
                    raise rv.ConfigError(
                        'ChipClassificationGeoJSONStoreConfig has no '
                        'URI set, and is not associated with a SceneConfig.')

            io_def.add_output(conf.uri)

        if command_type == rv.EVAL:
            if self.uri:
                io_def.add_input(self.uri)
            else:
                msg = 'No URI set for ChipClassificationGeoJSONStoreConfig'
                io_def.add_missing(msg)

        return (conf, io_def)


class ChipClassificationGeoJSONStoreConfigBuilder(LabelStoreConfigBuilder):
    def __init__(self, prev=None):
        config = {}
        if prev:
            config = {'uri': prev.uri}

        super().__init__(ChipClassificationGeoJSONStoreConfig, config)

    def from_proto(self, msg):
        b = ChipClassificationGeoJSONStoreConfigBuilder()

        return b \
            .with_uri(msg.uri)

    def with_uri(self, uri):
        """Set URI for a GeoJSON used to read/write predictions."""
        b = deepcopy(self)
        b.config['uri'] = uri
        return b
