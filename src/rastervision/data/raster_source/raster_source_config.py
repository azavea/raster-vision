from abc import abstractmethod
from copy import deepcopy

import rastervision as rv
from rastervision.core.config  import (Config, ConfigBuilder)
from rastervision.data import (RasterTransformerConfig,
                               StatsTransformerConfig)
from rastervision.protos.raster_source2_pb2 import RasterSourceConfig as RasterSourceConfigMsg

class RasterSourceConfig(Config):
    def __init__(self,
                 source_type,
                 transformers=None,
                 channel_order=None):
        if transformers is None:
            transformers = []

        self.source_type = source_type
        self.transformers = transformers
        self.channel_order = channel_order

    def to_proto(self):
        transformers = list(map(lambda c: c.to_proto(),
                                    self.transformers))
        msg = RasterSourceConfigMsg(source_type=self.source_type,
                                    channel_order=self.channel_order,
                                    transformers=transformers)
        return msg

    @abstractmethod
    def create_source(self, tmp_dir):
        """Create the Raster Source for this configuration.
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.RASTER_SOURCE,
                                               self.source_type)(self)

    @staticmethod
    def builder(source_type):
        return rv._registry.get_config_builder(rv.RASTER_SOURCE,
                                               source_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.RASTER_SOURCE, msg.source_type)() \
                           .from_proto(msg) \
                           .build()

    def create_transformers(self):
        return list(map(lambda c: c.create_transformer(), self.transformers))

    def preprocess_command(self, command_type, experiment_config, context=None):
        io_def = rv.core.CommandIODefinition()
        new_transformers = []
        for transformer in self.transformers:
            t, sub_io_def = transformer.preprocess_command(command_type, experiment_config, context)
            new_transformers.append(t)
            io_def.merge(sub_io_def)

        conf = self.to_builder() \
                   .with_transformers(new_transformers) \
                   .build()

        return (conf, io_def)


class RasterSourceConfigBuilder(ConfigBuilder):
    def from_proto(self, msg):
        transformers = list(map(lambda m: RasterTransformerConfig.from_proto(m),
                                msg.transformers))

        return self.with_channel_order(msg.channel_order) \
                   .with_transformers(transformers)

    # TODO: Standardize on deepcopy'ing the builder, not the config.
    # This allows parent builders to return child builder types.
    def with_channel_order(self, channel_order):
        """Defines the channel order for this raster source.

        Args:
            channel_order: numpy array of length n where n is the number of
                channels to use and the values are channel indices
        """
        b = deepcopy(self)
        b.config['channel_order'] = channel_order
        return b

    def with_transformers(self, transformers):
        b = deepcopy(self)
        b.config['transformers'] = transformers
        return b

    def with_stats_transformer(self):
        b = deepcopy(self)
        transformers = b.config.get('transformers')
        if transformers:
            b.config['transformers'] = transformers.append(StatsTransformerConfig())
        else:
            b.config['transformers'] = [StatsTransformerConfig()]
        return b
