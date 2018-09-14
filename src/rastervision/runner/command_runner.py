from tempfile import TemporaryDirectory

import click

import rastervision as rv
from rastervision.utils.files import load_json_config
from rastervision.protos.command_pb2 import CommandConfig as CommandConfigMsg


@click.command
@click.argument('command_config_uri')
def run(command_config_uri):
    with TemporaryDirectory as tmp_dir:
        msg = load_json_config(command_config_uri, CommandConfigMsg())
        command_config = rv.CommandConfig.from_proto(msg)
        command = command_config.create_command(tmp_dir)
        command.run(tmp_dir)


if __name__ == '__main__':
    run()
