from abc import abstractmethod
import logging

import rastervision as rv
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.protos.label_source_pb2 import LabelSourceConfig as LabelSourceConfigMsg

log = logging.getLogger(__name__)


class LabelSourceConfig(Config):
    deprecation_warnings = []

    def __init__(self, source_type):
        self.source_type = source_type

    def to_proto(self):
        msg = LabelSourceConfigMsg()
        msg.source_type = self.source_type
        return msg

    @abstractmethod
    def create_source(self, task_config, extent, crs_transformer, tmp_dir):
        """Create the Label Source for this configuration.

           Args:
              task_config: The TaskConfig for which this label source is supplying labels.
              extent: extent that this label source is applicable for
              crs_transformer: The crs_transformer used by the raster source this
                               label source is describing.
              tmp_dir: The temporary directory to use if files will need to be downloaded,
                       or None if only using local files.
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.LABEL_SOURCE,
                                               self.source_type)(self)

    @staticmethod
    def check_deprecation(source_type):
        # If source_type is deprecated and warning hasn't been shown yet, then warn.
        if (source_type in rv.label_source_deprecated_map
                and source_type not in LabelSourceConfig.deprecation_warnings):
            LabelSourceConfig.deprecation_warnings.append(source_type)
            new_source_type = rv.label_source_deprecated_map[source_type]
            log.warn(
                'LabelSource {} is deprecated. Please use {} instead.'.format(
                    source_type, new_source_type))

    @staticmethod
    def builder(source_type):
        LabelSourceConfig.check_deprecation(source_type)
        return rv._registry.get_config_builder(rv.LABEL_SOURCE, source_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a config from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.LABEL_SOURCE, msg.source_type)() \
                           .from_proto(msg) \
                           .build()


class LabelSourceConfigBuilder(ConfigBuilder):
    pass
