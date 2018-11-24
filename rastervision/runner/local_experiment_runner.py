import subprocess

import logging

from rastervision.runner import AwsBatchExperimentRunner

log = logging.getLogger(__name__)


def shellout(command_type,
             experiment_id,
             job_queue,
             job_definition,
             command,
             attempts=None,
             parent_job_ids=None,
             array_size=None):
    return subprocess.call(command, shell=True)


class LocalExperimentRunner(AwsBatchExperimentRunner):
    def __init__(self, tmp_dir=None):
        super().__init__()
        self.submit = shellout
        self.execution_environment = 'Shell'
        self.tmp_dir = tmp_dir
