from tempfile import TemporaryDirectory

import rastervision as rv
from rastervision.plugin import PluginRegistry
from rastervision.utils.files import load_json_config
from rastervision.protos.command_pb2 import CommandConfig as CommandConfigMsg


class CommandRunner:
    @staticmethod
    def run(command_config_uri):
        msg = load_json_config(command_config_uri, CommandConfigMsg())
        CommandRunner.run_from_proto(msg)

    def run_from_proto(msg):
        with TemporaryDirectory() as tmp_dir:
            PluginRegistry.get_instance().add_plugins_from_proto(msg.plugins)
            command_config = rv.command.CommandConfig.from_proto(msg)
            command = command_config.create_command(tmp_dir)
            command.run(tmp_dir)
