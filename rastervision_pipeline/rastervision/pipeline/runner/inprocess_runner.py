from typing import List
from rastervision.pipeline.cli import _run_command
from rastervision.pipeline.runner.runner import Runner

INPROCESS = 'inprocess'


class InProcessRunner(Runner):
    """Runs each command sequentially within a single process.

    Useful for testing and debugging.
    """

    def run(self,
            cfg_json_uri,
            pipeline,
            commands,
            num_splits=1,
            pipeline_run_name: str = 'raster-vision'):

        for command in commands:
            if command in pipeline.split_commands and num_splits > 1:
                for split_ind in range(num_splits):
                    _run_command(cfg_json_uri, command, split_ind, num_splits)
            else:
                _run_command(cfg_json_uri, command, 0, 1)

    def run_command(self, cmd: List[str]):
        raise NotImplementedError(
            'Use LocalRunner.run_command to run a command locally.')
