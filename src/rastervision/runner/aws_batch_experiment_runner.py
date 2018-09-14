import os

from rastervision.runner import ExperimentRunner
from rastervision.utils.batch import _batch_submit
from rastervision.utils.files import save_json_config


def make_command(command_config_uri):
    return 'python -m rastervision.runner.command_runner {}'.format(
        command_config_uri)


class AwsBatchExperimentRunner(ExperimentRunner):
    def __init__(self):
        # TODO: Fill this out from configuration
        self.branch = 'develop'
        self.attemps = 1
        self.gpu = True

    def _run_experiment(self, command_dag):
        """Runs all commands on AWS Batch."""

        ids_to_job = {}
        for command_id in command_dag.get_sorted_command_ids():
            command_config = command_dag.get_command(command_id)
            command_root_uri = command_config.root_uri
            command_uri = os.path.join(command_root_uri, 'command-config.json')
            save_json_config(command_config.to_proto(), command_uri)

            parent_job_ids = []
            for upstream_id in command_dag.get_upstream_command_ids(
                    command_id):
                if upstream_id not in ids_to_job:
                    raise Exception(
                        '{} command has parent command of {}, '
                        'but does not exist in previous batch submissions - '
                        'topological sort on command_dag error.')
                parent_job_ids.append(ids_to_job(upstream_id))

            batch_run_command = make_command(command_uri)
            job_id = _batch_submit(
                self.branch,
                batch_run_command,
                attempts=self.attempts,
                gpu=self.gpu,
                parent_job_ids=parent_job_ids)

            ids_to_job[command_id] = job_id
