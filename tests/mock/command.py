import os
import unittest
from unittest.mock import Mock

from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.command import (Command, CommandConfig, CommandConfigBuilder)

MOCK_COMMAND = 'MOCK_COMMAND'


class MockCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config
        self.mock = Mock()

    def run(self, tmp_dir=None):
        self.mock.run(tmp_dir)


class MockCommandConfig(CommandConfig):
    def __init__(self, root_uri):
        super().__init__(MOCK_COMMAND, root_uri)
        self.mock = Mock()

        self.mock.to_proto.return_value = None
        self.mock.create_command.return_value = None
        self.mock.update_for_command.return_value = None
        self.mock.report_io.return_value = None

    def create_command(self, tmp_dir=None):
        result = self.mock.to_proto()
        if result is None:
            retval = MockCommand(self)
            retval.set_tmp_dir(tmp_dir)
            return retval
        else:
            return result

    def to_proto(self):
        result = self.mock.to_proto()
        if result is None:
            msg = super().to_proto()
            msg.MergeFrom(CommandConfigMsg(custom_config=struct_pb2.Struct()))

            return msg
        else:
            return result

    def report_io(self):
        result = self.mock.to_proto()
        if result is None:
            return rv.core.CommandIODefinition()
        else:
            return result


class MockCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(prev)

    def validate(self):
        super().validate()

    def build(self):
        self.validate()
        return MockCommandConfig(self.root_uri)

    def from_proto(self, msg):
        b = super().from_proto(msg)

        conf = msg.custom_config

        return b

    def get_root_uri(self, experiment_config):
        mock_key = experiment_config.custom_config.get('mock_key')
        if not mock_key:
            mock_uri = experiment_config.custom_config.get('mock_uri')
            if not mock_uri:
                raise rv.ConfigError('MockCommand requires a mock_key or mock_uri '
                                     'be set in the experiment custom_config')
        else:
            mock_uri = os.path.join(experiment_config.root_uri, "mock", mock_key)

        return mock_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        return b
