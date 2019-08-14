import os

from google.protobuf import struct_pb2

import rastervision as rv
from rastervision.command import (Command, CommandConfig, CommandConfigBuilder)
from rastervision.protos.command_pb2 \
    import CommandConfig as CommandConfigMsg

NOOP_COMMAND = 'NOOP_COMMAND'


class NoopCommand(Command):
    def __init__(self, command_config):
        self.command_config = command_config

    def run(self, tmp_dir=None):
        pass


class NoopCommandConfig(CommandConfig):
    def __init__(self, root_uri):
        super().__init__(NOOP_COMMAND, root_uri)

    def create_command(self, tmp_dir=None):
        retval = NoopCommand(self)
        retval.set_tmp_dir(tmp_dir)
        return retval

    def to_proto(self):
        msg = super().to_proto()
        msg.MergeFrom(CommandConfigMsg(custom_config=struct_pb2.Struct()))

        return msg

    def report_io(self):
        return rv.core.CommandIODefinition()


class NoopCommandConfigBuilder(CommandConfigBuilder):
    def __init__(self, command_type, prev=None):
        super().__init__(command_type, prev)

    def validate(self):
        super().validate()

    def build(self):
        self.validate()
        return NoopCommandConfig(self.root_uri)

    def from_proto(self, msg):
        return super().from_proto(msg)

    def get_root_uri(self, experiment_config):
        noop_key = experiment_config.custom_config.get('noop_key')
        if not noop_key:
            noop_uri = experiment_config.custom_config.get('noop_uri')
            if not noop_uri:
                raise rv.ConfigError(
                    'NoopCommand requires a noop_key or noop_uri '
                    'be set in the experiment custom_config')
        else:
            noop_uri = os.path.join(experiment_config.root_uri, 'noop',
                                    noop_key)

        return noop_uri

    def with_experiment(self, experiment_config):
        b = super().with_experiment(experiment_config)
        return b


def register_plugin(plugin_registry):
    plugin_registry.register_command_config_builder(NOOP_COMMAND,
                                                    NoopCommandConfigBuilder)
