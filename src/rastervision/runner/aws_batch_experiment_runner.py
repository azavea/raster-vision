import os
import uuid
import click

import boto3

from rastervision.rv_config import RVConfig
from rastervision.runner import ExperimentRunner
from rastervision.utils.files import save_json_config


def make_command(command_config_uri):
    command = 'python -m rastervision run_command {}'.format(
        command_config_uri)
    return command


def batch_submit(command_type,
                 job_queue,
                 job_definition,
                 branch_name,
                 command,
                 attempts=3,
                 gpu=False,
                 parent_job_ids=None,
                 array_size=None):
    """
        Submit a job to run on Batch.

        Args:
            branch_name: Branch with code to run on Batch
            command: Command in quotes to run on Batch
    """
    if parent_job_ids is None:
        parent_job_ids = []

    full_command = command.split()

    client = boto3.client('batch')

    job_name = '{}_{}'.format(command_type, uuid.uuid4())
    depends_on = [{'jobId': job_id} for job_id in parent_job_ids]

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_definition,
        'containerOverrides': {
            'command': full_command
        },
        'retryStrategy': {
            'attempts': attempts
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


class AwsBatchExperimentRunner(ExperimentRunner):
    def __init__(self):
        rv_config = RVConfig.get_instance()

        batch_config = rv_config.get_subconfig('AWS_BATCH')

        self.branch = batch_config('branch', default='develop')
        self.attempts = batch_config('attempts', parser=int, default='1')
        self.gpu = batch_config('gpu', parser=bool, default='true')

        job_queue = batch_config('job_queue', default='')
        if not job_queue:
            if self.gpu:
                job_queue = 'raster-vision-gpu'
            else:
                job_queue = 'raster-vision-cpu'
        self.job_queue = job_queue

        job_definition = batch_config('job_definition', default='')
        if not job_definition:
            if self.gpu:
                job_definition = 'raster-vision-gpu'
            else:
                job_definition = 'raster-vision-cpu'
        self.job_definition = job_definition

    def _run_experiment(self, command_dag):
        """Runs all commands on AWS Batch."""

        ids_to_job = {}
        for command_id in command_dag.get_sorted_command_ids():
            command_config = command_dag.get_command(command_id)
            command_root_uri = command_config.root_uri
            command_uri = os.path.join(command_root_uri, 'command-config.json')
            print('Saving command configuration to {}...'.format(command_uri))
            save_json_config(command_config.to_proto(), command_uri)

            parent_job_ids = []
            for upstream_id in command_dag.get_upstream_command_ids(
                    command_id):
                if upstream_id not in ids_to_job:
                    cur_command = (command_config.command_type, command_id)
                    u = command_dag.get_command(upstream_id)
                    upstream_command = (u.command_type, upstream_id)
                    raise Exception(
                        '{} command has parent command of {}, '
                        'but does not exist in previous batch submissions - '
                        'topological sort on command_dag error.'.format(
                            cur_command, upstream_command))
                parent_job_ids.append(ids_to_job[upstream_id])

            batch_run_command = make_command(command_uri)
            job_id = batch_submit(
                command_config.command_type,
                self.job_queue,
                self.job_definition,
                self.branch,
                batch_run_command,
                attempts=self.attempts,
                gpu=self.gpu,
                parent_job_ids=parent_job_ids)

            ids_to_job[command_id] = job_id
