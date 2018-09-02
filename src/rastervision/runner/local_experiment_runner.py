from tempfile import TemporaryDirectory

from rastervision.runner import ExperimentRunner

class LocalExperimentRunner(ExperimentRunner):
    def __init__(self, tmp_dir=None):
        self.tmp_dir = tmp_dir
    def _run_experiment(self, command_dag):
        """Runs all commands on this machine."""

        def run_commands(tmp_dir):
            for command_config in command_dag.get_sorted_commands():
                command = command_config.create_command(tmp_dir)
                command.run(tmp_dir)

        if self.tmp_dir:
            run_commands(self.tmp_dir)
        else:
            with TemporaryDirectory() as tmp_dir:
                run_commands(tmp_dir)
