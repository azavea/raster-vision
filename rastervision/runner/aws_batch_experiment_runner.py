import uuid
import click

from rastervision.runner import OutOfProcessExperimentRunner
from rastervision.rv_config import RVConfig


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
                     experiment_id,
                     command,
                     parent_job_ids=None,
                     array_size=None):
        """
        Submit a job to run on Batch.

        Args:
           command_type: (str) the type of command. ie. a value in rv.command.api
           experiment_id: (str) id of experiment
           command: (str) command to run inside Docker container
           parent_job_ids (list of str): ids of jobs that this job depends on
           array_size: (int) size of the Batch array job
        """
        import boto3

        if parent_job_ids is None:
            parent_job_ids = []

        full_command = command.split()

        client = boto3.client('batch')

        uuid_part = str(uuid.uuid4()).split('-')[0]
        exp = ''.join(e for e in experiment_id if e.isalnum())
        job_name = '{}_{}_{}'.format(command_type, exp, uuid_part)
        depends_on = [{'jobId': job_id} for job_id in parent_job_ids]

        kwargs = {
            'jobName': job_name,
            'jobQueue': self.job_queue,
            'jobDefinition': self.job_definition,
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

        msg = '{} command submitted job with jobName={} and jobId={}'.format(
            command_type, job_name, job_id)
        click.echo(click.style(msg, fg='green'))

        return job_id
