import subprocess
from inspect import signature

from rastervision2.pipeline.cli import _run_command
from rastervision2.pipeline.runner.runner import Runner

INPROCESS = 'inprocess'


class InProcessRunner(Runner):
    """Runs each command sequentially within a single process.

    Useful for testing and debugging.
    """

    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        for command in commands:

            # detect external command
            if hasattr(pipeline, command):
                fn = getattr(pipeline, command)
                params = signature(fn).parameters
                external = hasattr(fn, 'external') and len(params) in {0, 1}
            else:
                external = False

            if not external:
                if command in pipeline.split_commands and num_splits > 1:
                    for split_ind in range(num_splits):
                        _run_command(cfg_json_uri, command, split_ind,
                                     num_splits)
                else:
                    _run_command(cfg_json_uri, command, 0, 1)
            else:
                if len(params) == 0:
                    # No-parameter external command
                    cmds = [fn()]
                elif len(params) == 1:
                    # One-paramater (split) external command
                    cmds = fn(num_splits)
                else:
                    # No command
                    cmds = []
                for cmd in cmds:
                    subprocess.run(cmd, check=True)
