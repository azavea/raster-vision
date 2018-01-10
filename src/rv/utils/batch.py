from os import environ

import click
import boto3

s3_bucket = environ.get('S3_BUCKET')


def _batch_submit(branch_name, command, attempts=3, cpu=False):
    """
        Submit a job to run on Batch.

        Args:
            branch_name: Branch with code to run on Batch
            command: Command in quotes to run on Batch
    """
    full_command = ['run_script.sh', branch_name]
    full_command.extend(command.split())

    client = boto3.client('batch')
    job_queue = 'raster-vision-cpu' if cpu else \
        'raster-vision-gpu'
    job_definition = 'raster-vision-cpu' if cpu else \
        'raster-vision-gpu'

    job_name = command.replace('/', '-').replace('.', '-')
    job_name = 'batch_submit'

    job_id = client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            'command': full_command
        },
        retryStrategy={
            'attempts': attempts
        })['jobId']

    click.echo(
        'Submitted job with jobName={} and jobId={}'.format(job_name, job_id))


@click.command()
@click.argument('branch_name')
@click.argument('command')
@click.option('--attempts', default=3, help='Number of times to retry job')
@click.option('--cpu', is_flag=True, help='Use CPU EC2 instances')
def batch_submit(branch_name, command, attempts, cpu):
    _batch_submit(branch_name, command, attempts=attempts, cpu=cpu)


if __name__ == '__main__':
    batch_submit()
