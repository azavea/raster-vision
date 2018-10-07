from abc import abstractmethod

import rastervision as rv
from rastervision.core.config import (Config, ConfigBuilder,
                                      BundledConfigMixin)


class AnalyzerConfig(BundledConfigMixin, Config):
    def __init__(self, analyzer_type):
        self.analyzer_type = analyzer_type

    @abstractmethod
    def create_analyzer(self):
        """Create the Transformer that this configuration represents
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.ANALYZER,
                                               self.analyzer_type)(self)

    @staticmethod
    def builder(analyzer_type):
        return rv._registry.get_config_builder(rv.ANALYZER, analyzer_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.ANALYZER, msg.analyzer_type)() \
                           .from_proto(msg) \
                           .build()


class AnalyzerConfigBuilder(ConfigBuilder):
    pass
