import rastervision as rv
from rastervision.plugin import PluginRegistry
from rastervision.protos.command_pb2 import CommandConfig as CommandConfigMsg
from rastervision.utils.files import load_json_config


class CommandRunner:
    @staticmethod
    def run(command_config_uri):
        msg = load_json_config(command_config_uri, CommandConfigMsg())
        CommandRunner.run_from_proto(msg)

    def run_from_proto(msg):
        PluginRegistry.get_instance().add_plugins_from_proto(msg.plugins)
        command_config = rv.command.CommandConfig.from_proto(msg)
        command = command_config.create_command()
        command.run()
