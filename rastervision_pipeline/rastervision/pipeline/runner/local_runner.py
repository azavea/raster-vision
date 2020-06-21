import sys
from inspect import signature
from os.path import dirname, join
from subprocess import Popen

from rastervision.pipeline.file_system import str_to_file
from rastervision.pipeline.runner.runner import Runner
from rastervision.pipeline.utils import terminate_at_exit

LOCAL = 'local'


class LocalRunner(Runner):
    """Runs each command locally using different processes for each command/split.

    This is implemented by generating a Makefile and then running it using make.
    """

    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
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

            # detect external command
            if hasattr(pipeline, command):
                fn = getattr(pipeline, command)
                params = signature(fn).parameters
                external = hasattr(fn, 'external') and len(params) in {0, 1}
            else:
                external = False

            curr_command_inds = []
            if not external:
                if command in pipeline.split_commands and num_splits > 1:
                    for split_ind in range(num_splits):
                        makefile += '{}: '.format(curr_command_ind)
                        makefile += ' '.join(
                            [str(ci) for ci in prev_command_inds])
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
            else:
                if len(params) == 0:
                    # No-parameter external command
                    cmds = [fn()]
                elif len(params) == 1 and command in pipeline.split_commands:
                    # One-paramater split external command
                    cmds = fn(num_splits)
                elif len(params
                         ) == 1 and command not in pipeline.split_commands:
                    # One-paramater unsplit external command
                    cmds = fn(1)
                else:
                    # No command
                    cmds = []
                for cmd in cmds:
                    makefile += '{}: '.format(curr_command_ind)
                    makefile += ' '.join([str(ci) for ci in prev_command_inds])
                    makefile += '\n'
                    invocation = (' '.join(cmd))
                    makefile += '\t{}\n\n'.format(invocation)
                    curr_command_inds.append(curr_command_ind)
                    curr_command_ind += 1

            prev_command_inds = curr_command_inds

        makefile_path = join(dirname(cfg_json_uri), 'Makefile')
        str_to_file(makefile, makefile_path)
        process = Popen(['make', '-j', '-f', makefile_path])
        terminate_at_exit(process)
        exitcode = process.wait()
        if exitcode != 0:
            sys.exit(exitcode)
        else:
            return 0
