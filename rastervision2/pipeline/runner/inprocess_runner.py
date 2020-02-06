from rastervision2.pipeline.main import _run_command
from rastervision2.pipeline.runner.runner import Runner

INPROCESS = 'inprocess'


class InProcessRunner(Runner):
    def run(self, cfg_json_uri, pipeline, commands, num_splits=1):
        for command in commands:
            if command in pipeline.split_commands and num_splits > 1:
                for split_ind in range(num_splits):
                    _run_command(cfg_json_uri, command, split_ind, num_splits)
            else:
                _run_command(cfg_json_uri, command, 0, 1)
