import rastervision as rv
from rastervision.data import (
    RasterTransformerConfig, RasterTransformerConfigBuilder,
    NoopTransformer)  # Already a real noop transformer
from rastervision.protos.raster_transformer_pb2 \
    import RasterTransformerConfig as RasterTransformerConfigMsg

NOOP_TRANSFORMER = 'NOOP_TRANSFORMER'


class NoopRasterTransformerConfig(RasterTransformerConfig):
    def __init__(self):
        super().__init__(NOOP_TRANSFORMER)

    def to_proto(self):
        msg = RasterTransformerConfigMsg(
            transformer_type=self.transformer_type, custom_config={})
        return msg

    def create_transformer(self):
        return NoopTransformer()

    def preprocess_command(self, command_type, experiment_config, context=[]):
        return (self, rv.core.CommandIODefinition())


class NoopRasterTransformerConfigBuilder(RasterTransformerConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopRasterTransformerConfig, {})

    def from_proto(self, msg):
        return self


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(
        rv.RASTER_TRANSFORMER, NOOP_TRANSFORMER,
        NoopRasterTransformerConfigBuilder)
