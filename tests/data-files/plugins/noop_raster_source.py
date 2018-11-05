import numpy as np

import rastervision as rv
from rastervision.core import Box
from rastervision.data import (RasterSource, RasterSourceConfig,
                               RasterSourceConfigBuilder,
                               IdentityCRSTransformer)
from rastervision.protos.raster_source_pb2 \
    import RasterSourceConfig as RasterSourceConfigMsg

NOOP_SOURCE = 'NOOP_SOURCE'


class NoopRasterSource(RasterSource):
    def get_extent(self):
        Box.make_square(0, 0, 2)

    def get_dtype(self):
        return np.uint8

    def get_crs_transformer(self):
        return IdentityCRSTransformer()

    def _get_chip(self, window):
        return np.random.rand(1, 2, 2, 3)


class NoopRasterSourceConfig(RasterSourceConfig):
    def __init__(self, transformers=None, channel_order=None):
        super().__init__(NOOP_SOURCE, transformers, channel_order)

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(RasterSourceConfigMsg(custom_config={}))
        return msg

    def create_source(self, tmp_dir):
        transformers = self.create_transformers()
        return NoopRasterSource(transformers, tmp_dir)

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        return io_def or rv.core.CommandIODefinition()

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self

    def for_prediction(self, image_uri):
        return self

    def create_local(self, tmp_dir):
        return self


class NoopRasterSourceConfigBuilder(RasterSourceConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopRasterSourceConfig, {})

    def from_proto(self, msg):
        return super().from_proto(msg)


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.RASTER_SOURCE, NOOP_SOURCE,
                                            NoopRasterSourceConfigBuilder)
