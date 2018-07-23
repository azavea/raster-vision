from os import environ
import uuid

import click
import boto3

s3_bucket = environ.get('S3_BUCKET')


def _batch_submit(branch_name,
                  command,
                  attempts=3,
                  gpu=False,
                  parent_job_ids=[],
                  array_size=None):
    """
        Submit a job to run on Batch.

        Args:
            branch_name: Branch with code to run on Batch
            command: Command in quotes to run on Batch
    """
    full_command = ['run_rv', branch_name]
    full_command.extend(command.split())

    client = boto3.client('batch')
    job_queue = 'raster-vision-gpu' if gpu else \
        'raster-vision-cpu'
    job_definition = 'raster-vision-gpu' if gpu else \
        'raster-vision-cpu'

    job_name = str(uuid.uuid4())
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

    click.echo('Submitted job with jobName={} and jobId={}'.format(
        job_name, job_id))

    return job_id


@click.command()
@click.argument('branch_name')
@click.argument('command')
@click.option('--attempts', default=3, help='Number of times to retry job')
@click.option('--gpu', is_flag=True, help='Use CPU EC2 instances')
def batch_submit(branch_name, command, attempts, gpu):
    _batch_submit(branch_name, command, attempts=attempts, gpu=gpu)


if __name__ == '__main__':
    batch_submit()
