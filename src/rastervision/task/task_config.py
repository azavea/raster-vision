from abc import abstractmethod
import os
from copy import deepcopy

import rastervision as rv
from rastervision.core import (Config, ConfigBuilder)
from rastervision.protos.task_pb2 import TaskConfig as TaskConfigMsg


class TaskConfig(Config):
    def __init__(self,
                 task_type,
                 predict_batch_size=10,
                 predict_package_uri=None,
                 debug=True):
        self.task_type = task_type
        self.predict_batch_size = predict_batch_size
        self.predict_package_uri = predict_package_uri
        self.debug = debug

    @abstractmethod
    def create_task(self, backend):
        """Create the Task that this configuration represents

           Args:
              backend: The backend to be used by the task.
        """
        pass

    def to_builder(self):
        return rv._registry.get_config_builder(rv.TASK, self.task_type)(self)

    def to_proto(self):
        return TaskConfigMsg(
            task_type=self.task_type,
            predict_batch_size=self.predict_batch_size,
            predict_package_uri=self.predict_package_uri,
            debug=self.debug)

    @staticmethod
    def builder(task_type):
        return rv._registry.get_config_builder(rv.TASK, task_type)()

    @staticmethod
    def from_proto(msg):
        """Creates a TaskConfig from the specificed protobuf message
        """
        return rv._registry.get_config_builder(rv.TASK, msg.task_type)() \
                           .from_proto(msg) \
                           .build()

    def preprocess_command(self, command_type, experiment_config,
                           context=None):
        conf = self
        if command_type == rv.BUNDLE:
            if not conf.predict_package_uri:
                conf.predict_package_uri = os.path.join(
                    experiment_config.bundle_uri, "predict_package.zip")
        return (conf, rv.core.CommandIODefinition())


class TaskConfigBuilder(ConfigBuilder):
    def with_predict_batch_size(self, predict_batch_size):
        """Sets the batch size to use during prediction."""
        b = deepcopy(self)
        b.config['predict_batch_size'] = predict_batch_size
        return b

    def with_predict_package_uri(self, predict_package_uri):
        """Sets the URI to save a predict package URI to during bundle."""
        b = deepcopy(self)
        b.config['predict_package_uri'] = predict_package_uri
        return b

    def with_debug(self, debug):
        """Flag for producing debug products."""
        b = deepcopy(self)
        b.config['debug'] = debug
        return b
