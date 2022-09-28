import sys
from os.path import dirname, join
from subprocess import Popen

from rastervision.pipeline.file_system import str_to_file, download_if_needed
from rastervision.pipeline.runner.runner import Runner
from rastervision.pipeline.utils import terminate_at_exit

LOCAL = 'local'


class LocalRunner(Runner):
    """Runs each command locally using different processes for each command/split.

    This is implemented by generating a Makefile and then running it using make.
    """

    def run(self,
            cfg_json_uri,
            pipeline,
            commands,
            num_splits=1,
            pipeline_run_name: str = 'raster-vision'):
        num_commands = 0
        for command in commands:
            if command in pipeline.split_commands and num_splits > 1:
                num_commands += num_splits
            else:
                num_commands += 1

        makefile = '.PHONY: '
        makefile += ' '.join([str(ci) for ci in range(num_commands)])
        makefile += '\n\n'

        makefile += 'all: '
        makefile += ' '.join([str(ci) for ci in range(num_commands)])
        makefile += '\n\n'

        prev_command_inds = []
        curr_command_ind = 0
        for command in commands:

            curr_command_inds = []
            if command in pipeline.split_commands and num_splits > 1:
                for split_ind in range(num_splits):
                    makefile += '{}: '.format(curr_command_ind)
                    makefile += ' '.join([str(ci) for ci in prev_command_inds])
                    makefile += '\n'
                    invocation = (
                        'python -m rastervision.pipeline.cli run_command '
                        '{} {} --split-ind {} --num-splits {}').format(
                            cfg_json_uri, command, split_ind, num_splits)
                    makefile += '\t{}\n\n'.format(invocation)
                    curr_command_inds.append(curr_command_ind)
                    curr_command_ind += 1
            else:
                makefile += '{}: '.format(curr_command_ind)
                makefile += ' '.join([str(ci) for ci in prev_command_inds])
                makefile += '\n'
                invocation = (
                    'python -m rastervision.pipeline.cli run_command '
                    '{} {}'.format(cfg_json_uri, command))
                makefile += '\t{}\n\n'.format(invocation)
                curr_command_inds.append(curr_command_ind)
                curr_command_ind += 1

            prev_command_inds = curr_command_inds

        makefile_path = join(dirname(cfg_json_uri), 'Makefile')
        str_to_file(makefile, makefile_path)
        makefile_path_local = download_if_needed(makefile_path)
        process = Popen(['make', '-j', '-f', makefile_path_local])
        terminate_at_exit(process)
        exitcode = process.wait()
        if exitcode != 0:
            sys.exit(exitcode)
        else:
            return 0
