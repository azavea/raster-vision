from subprocess import Popen
import sys
import logging

from rastervision.runner import OutOfProcessExperimentRunner
from rastervision.utils.misc import (terminate_at_exit)

log = logging.getLogger(__name__)


class LocalExperimentRunner(OutOfProcessExperimentRunner):
    def __init__(self, tmp_dir=None):
        super().__init__()

        self.submit = self.shellout
        self.execution_environment = 'Shell'
        self.tmp_dir = tmp_dir

    def shellout(self,
                 command_type,
                 experiment_id,
                 command,
                 parent_job_ids=None,
                 array_size=None):
        command = list(filter(lambda s: len(s) > 0, command.split(' ')))
        process = Popen(command)
        terminate_at_exit(process)
        exitcode = process.wait()
        if exitcode != 0:
            sys.exit(exitcode)
        else:
            return 0
