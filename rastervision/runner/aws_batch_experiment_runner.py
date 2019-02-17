import uuid
import click
import logging

from rastervision.runner import OutOfProcessExperimentRunner
from rastervision.rv_config import RVConfig
from rastervision.utils.misc import grouped
import rastervision as rv

DEPENDENCY_GROUP_JOB = 'DEPENDENCY_GROUP'

CPU_STAGES = {rv.ANALYZE, rv.CHIP, rv.EVAL, rv.BUNDLE, DEPENDENCY_GROUP_JOB}

NOOP_COMMAND = 'python -m rastervision --help'

log = logging.getLogger(__name__)


class AwsBatchExperimentRunner(OutOfProcessExperimentRunner):
    def __init__(self):
        super().__init__()

        rv_config = RVConfig.get_instance()

        batch_config = rv_config.get_subconfig('AWS_BATCH')

        self.attempts = batch_config('attempts', parser=int, default='1')
        self.gpu = batch_config('gpu', parser=bool, default='true')

        job_queue = batch_config('job_queue', default='')
        if not job_queue:
            if self.gpu:
                job_queue = 'raster-vision-gpu'
            else:
                job_queue = 'raster-vision-cpu'
        self.job_queue = job_queue

        cpu_job_queue = batch_config('cpu_job_queue', default='')
        if not cpu_job_queue:
            if self.gpu:
                cpu_job_queue = 'raster-vision-cpu'
            else:
                cpu_job_queue = job_queue
        self.cpu_job_queue = cpu_job_queue

        job_definition = batch_config('job_definition', default='')
        if not job_definition:
            if self.gpu:
                job_definition = 'raster-vision-gpu'
            else:
                job_definition = 'raster-vision-cpu'
        self.job_definition = job_definition

        cpu_job_definition = batch_config('cpu_job_definition', default='')
        if not cpu_job_definition:
            if self.gpu:
                cpu_job_definition = 'raster-vision-cpu'
            else:
                cpu_job_definition = job_definition
        self.cpu_job_definition = cpu_job_definition

        self.submit = self.batch_submit
        self.execution_environment = 'Batch'

    def batch_submit(self,
                     command_type,
                     command_split_id,
                     experiment_id,
                     command,
                     parent_job_ids=None,
                     array_size=None):
        """
        Submit a job to run on Batch.

        Args:
           command_type: (str) the type of command. ie. a value in rv.command.api
           command_split_id: (int or str) the split ID of command.
                             ie. the parallelized command ID
           experiment_id: (str) id of experiment
           command: (str) command to run inside Docker container
           parent_job_ids (list of str): ids of jobs that this job depends on
           array_size: (int) size of the Batch array job
        """
        import boto3

        if parent_job_ids is None:
            parent_job_ids = []

        if command_type == DEPENDENCY_GROUP_JOB:
            full_command = NOOP_COMMAND.split()
        else:
            full_command = command.split()

        client = boto3.client('batch')

        uuid_part = str(uuid.uuid4()).split('-')[0]
        exp = ''.join(e for e in experiment_id if e.isalnum())
        job_name = '{}-{}_{}_{}'.format(command_type, command_split_id, exp,
                                        uuid_part)

        group_level = 1
        while len(parent_job_ids) > 20:
            # AWS Batch only allows for 20 parent jobs.
            # This hacks around that limit.
            # Group dependencies  into batches of 20,
            # and submit noop jobs that form a graph
            # of dependencies where no set of leaves is
            # greater than 20.
            log.warn('More that 20 parent jobs detected, grouping [LEVEL {}]'.
                     format(group_level))
            new_parents = []
            group_id = str(uuid.uuid4()).split('-')[0]
            for i, group in enumerate(grouped(parent_job_ids, 20)):
                group_split_id = '{}_{}-{}_{}'.format(group_id, command_type,
                                                      command_split_id, i)
                new_parents.append(
                    self.batch_submit(DEPENDENCY_GROUP_JOB, group_split_id,
                                      experiment_id, NOOP_COMMAND, group,
                                      array_size))
            parent_job_ids = new_parents
            group_level += 1

        depends_on = [{'jobId': job_id} for job_id in parent_job_ids]

        if command_type in CPU_STAGES:
            job_queue = self.cpu_job_queue
            job_definition = self.cpu_job_definition
        else:
            job_queue = self.job_queue
            job_definition = self.job_definition

        kwargs = {
            'jobName': job_name,
            'jobQueue': job_queue,
            'jobDefinition': job_definition,
            'containerOverrides': {
                'command': full_command
            },
            'retryStrategy': {
                'attempts': self.attempts
            },
            'dependsOn': depends_on
        }

        if array_size is not None:
            kwargs['arrayProperties'] = {'size': array_size}

        job_id = client.submit_job(**kwargs)['jobId']

        msg = '{}-{} command submitted job with jobName={} and jobId={}'.format(
            command_type, command_split_id, job_name, job_id)
        click.echo(click.style(msg, fg='green'))

        return job_id
