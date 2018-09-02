from tempfile import TemporaryDirectory

from rastervision.runner import ExperimentRunner
from rastervision.utils.batch import _batch_submit

def make_command(command_config_uri):
    return 'python -m rastervision.runner.command_runner {}'.format(command_config_uri)

class AwsBatchExperimentRunner(ExperimentRunner):
    def _run_experiment(self, commands_dag):
        """Runs all commands on AWS Batch."""

        idx_to_job  = {}
        for command_config in command_dag.get_sorted_commands():
            command_root_uri = command_config.root_uri
            command_uri = os.path.join(command_root_uri, "command-config.json")
            save_json_config(command_config.to_proto(), command_uri)

            parent_job_ids = []
            for parent_idx in command_dag.in_edges(idx):
                if parent_idx not in idx_to_job:
                    raise Exception("{} command has parent command of {}, "
                                    "but does not exist in previous batch submissions - "
                                    "topological sort on command_dag error.")
                parent_job_ids.append(idx_to_job(parent_idx))

            batch_run_command = make_command(command_uri)
            job_id = _batch_submit(branch, batch_run_command, attempts=1, gpu=True)

            idx_to_job[idx] = job_id
