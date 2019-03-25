import os

import logging

from rastervision.runner import ExperimentRunner
from rastervision.utils.files import save_json_config
from rastervision.rv_config import RVConfig
import rastervision as rv

log = logging.getLogger(__name__)


class InProcessExperimentRunner(ExperimentRunner):
    """A class implementing functionality for running experiments within
    the present process.

    Note that the bulk of some stages still run in different process,
    as for example when a separate training script is run as a shell
    command.  The contrast between this behavior and the
    out-of-process runners (those derived from
    `OutOfProcessExperimentRunner`) is that in the latter, those shell
    commands are run from stage-specific processes.

    """

    def __init__(self, tmp_dir=None):
        self.tmp_dir = tmp_dir

    def _run_experiment(self, command_dag):
        """Runs all commands on this machine."""

        def run_commands(tmp_dir):
            for command_config in command_dag.get_sorted_commands():
                msg = command_config.to_proto()
                builder = rv._registry.get_command_config_builder(
                    msg.command_type)()
                command_config = builder.from_proto(msg).build()

                command_root_uri = command_config.root_uri
                command_basename = 'command-config-{}.json'.format(
                    command_config.split_id)
                command_uri = os.path.join(command_root_uri, command_basename)
                log.info('Saving command configuration to {}...'.format(
                    command_uri))
                save_json_config(command_config.to_proto(), command_uri)

                command = command_config.create_command()
                command.run(tmp_dir)

        if self.tmp_dir:
            run_commands(self.tmp_dir)
        else:
            with RVConfig.get_tmp_dir() as tmp_dir:
                run_commands(tmp_dir)
