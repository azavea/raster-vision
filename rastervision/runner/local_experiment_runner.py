from subprocess import Popen
import sys
import logging
import os

from rastervision.runner import OutOfProcessExperimentRunner
from rastervision.runner import make_command
from rastervision.utils.misc import (terminate_at_exit)
from rastervision.utils.files import (save_json_config, make_dir)
from rastervision.rv_config import RVConfig

log = logging.getLogger(__name__)


class LocalExperimentRunner(OutOfProcessExperimentRunner):
    def __init__(self, tmp_dir=None):
        super().__init__()

        self.submit = None
        self.execution_environment = 'Shell'
        self.tmp_dir = tmp_dir

    def _run_experiment(self, command_dag):
        tmp_dir = self.tmp_dir or RVConfig.get_tmp_dir().name
        make_dir(tmp_dir)
        makefile_name = os.path.join(tmp_dir, 'Makefile')
        with open(makefile_name, 'w') as makefile:
            command_ids = command_dag.get_sorted_command_ids()

            # .PHONY: 0 1 2 3 4 5
            makefile.write('.PHONY:')
            for command_id in command_ids:
                makefile.write(' {}'.format(command_id))
            makefile.write('\n\n')

            # all: 0 1 2 3 4 5
            makefile.write('all:')
            for command_id in command_ids:
                makefile.write(' {}'.format(command_id))
            makefile.write('\n\n')

            for command_id in command_ids:
                # 0: 1 2
                makefile.write('{}:'.format(command_id))
                for upstream_id in command_dag.get_upstream_command_ids(
                        command_id):
                    makefile.write(' {}'.format(upstream_id))
                makefile.write('\n')

                # \t rastervision ...
                command_def = command_dag.get_command_definition(command_id)
                command_config = command_def.command_config
                command_root_uri = command_config.root_uri
                command_basename = 'command-config-{}.json'.format(
                    command_config.split_id)
                command_uri = os.path.join(command_root_uri, command_basename)
                print('Saving command configuration to {}...'.format(
                    command_uri))
                save_json_config(command_config.to_proto(), command_uri)
                run_command = make_command(command_uri, self.tmp_dir)
                makefile.write('\t{}\n\n'.format(run_command))

        process = Popen(['make', '-j', '-f', makefile_name])
        terminate_at_exit(process)
        exitcode = process.wait()
        if exitcode != 0:
            sys.exit(exitcode)
        else:
            return 0
