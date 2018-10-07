from abc import abstractmethod

import rastervision as rv
from rastervision.core.config import (Config, ConfigBuilder)
from rastervision.protos.label_store_pb2 import LabelStoreConfig as LabelStoreConfigMsg


class LabelStoreConfig(Config):
    def __init__(self, store_type):
        self.store_type = store_type

    def to_proto(self):
        msg = LabelStoreConfigMsg()
        msg.store_type = self.store_type
        return msg

    @abstractmethod
    def for_prediction(self, label_uri):
        """Build a copy of this config with a label_uri set.

        This is used in the Predictor to save labels to a specific path.
        """
        pass

    @abstractmethod
    def create_store(self, task_config, extent, crs_transformer, tmp_dir):
        """Create the Label Store for this configuration.

           Args:
              task_config: The TaskConfig for which this label source is supplying labels.
              crs_transformer: The crs_transformer used by the raster store this
                               label store is describing.
              tmp_dir: The temporary directory to use if files will need to be downloaded,
                       or None if only using local files.
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.LABEL_STORE,
                                               self.store_type)(self)

    @staticmethod
    def builder(store_type):
        return rv._registry.get_config_builder(rv.LABEL_STORE, store_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.LABEL_STORE, msg.store_type)() \
                           .from_proto(msg) \
                           .build()


class LabelStoreConfigBuilder(ConfigBuilder):
    pass
