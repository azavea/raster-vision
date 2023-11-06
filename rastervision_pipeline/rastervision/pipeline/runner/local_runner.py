from typing import TYPE_CHECKING, List, Optional
import sys
from os.path import dirname, join
from subprocess import Popen

from rastervision.pipeline.file_system import str_to_file, download_if_needed
from rastervision.pipeline.runner.runner import Runner
from rastervision.pipeline.utils import terminate_at_exit

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline import Pipeline

LOCAL = 'local'


def make_run_cmd_invocation(cfg_json_uri: str,
                            command: str,
                            opts: Optional[dict] = None) -> str:
    opts_str = ''
    if opts is not None:
        opts_str = ' ' + ' '.join(f'{k} {v}' for k, v in opts.items())
    return ('python -m rastervision.pipeline.cli run_command '
            f'{cfg_json_uri} {command}{opts_str}')


def make_makefile_entry_for_cmd(curr_command_ind: int,
                                prev_command_inds: List[int],
                                cfg_json_uri: str,
                                command: str,
                                opts: Optional[dict] = None) -> str:
    out = f'{curr_command_ind}: '
    out += ' '.join([str(ci) for ci in prev_command_inds])
    out += '\n'
    invocation = make_run_cmd_invocation(cfg_json_uri, command, opts=opts)
    out += f'\t{invocation}\n\n'
    return out


class LocalRunner(Runner):
    """
    Runs each command locally using different processes for each command/split.

    This is implemented by generating a Makefile and then running it using
    make.
    """

    def run(self,
            cfg_json_uri: str,
            pipeline: 'Pipeline',
            commands: List[str],
            num_splits: int = 1,
            pipeline_run_name: str = 'raster-vision'):
        makefile = self.build_makefile_string(cfg_json_uri, pipeline, commands,
                                              num_splits)
        makefile_path = join(dirname(cfg_json_uri), 'Makefile')
        str_to_file(makefile, makefile_path)
        makefile_path_local = download_if_needed(makefile_path)
        return self.run_command(['make', '-j', '-f', makefile_path_local])

    def run_command(self, cmd: List[str]):
        process = Popen(cmd)
        terminate_at_exit(process)
        exitcode = process.wait()
        if exitcode != 0:
            sys.exit(exitcode)
        else:
            return 0

    def build_makefile_string(self,
                              cfg_json_uri: str,
                              pipeline: 'Pipeline',
                              commands: List[str],
                              num_splits: int = 1) -> str:
        num_commands = 0
        for command in commands:
            if command in pipeline.split_commands and num_splits > 1:
                num_commands += num_splits
            else:
                num_commands += 1

        all_command_inds_str = ' '.join([str(i) for i in range(num_commands)])
        makefile = f'.PHONY: {all_command_inds_str}\n\n'
        makefile += f'all: {all_command_inds_str}\n\n'

        prev_command_inds = []
        curr_command_ind = 0
        for command in commands:
            curr_command_inds = []
            if command in pipeline.split_commands and num_splits > 1:
                opts = {'--num-splits': num_splits}
                for split_ind in range(num_splits):
                    opts['--split-ind'] = split_ind
                    makefile_entry = make_makefile_entry_for_cmd(
                        curr_command_ind,
                        prev_command_inds,
                        cfg_json_uri,
                        command,
                        opts=opts)
                    makefile += makefile_entry
                    curr_command_inds.append(curr_command_ind)
                    curr_command_ind += 1
            else:
                makefile_entry = make_makefile_entry_for_cmd(
                    curr_command_ind, prev_command_inds, cfg_json_uri, command)
                makefile += makefile_entry
                curr_command_inds.append(curr_command_ind)
                curr_command_ind += 1
            prev_command_inds = curr_command_inds
        return makefile
