from os import environ
import uuid
import subprocess

import click
import boto3


def is_branch_valid(repo, branch):
    ls_branch_command = [
        'git', 'ls-remote', '--heads', repo, branch]

    if not subprocess.run(ls_branch_command, stdout=subprocess.PIPE).stdout:
        print('Error: remote branch {} does not exist'.format(branch))
        return False
    return True


def _batch_submit(command, repo=repo, branch=branch, attempts=3, gpu=False,
                  parent_job_ids=[], array_size=None):
    """
        Submit a job to run on Batch.

        Args:
            repo: URI for Github repo with code to run
            branch: Branch with code to run
            command: Command in quotes to run
    """
    full_command = ['run_rv', repo, branch]
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

    click.echo(
        'Submitted job with jobName={} and jobId={}'.format(job_name, job_id))

    return job_id


@click.command()
@click.argument('command')
@click.option('--repo', default='https://github.com/azavea/raster-vision.git')
@click.option('--branch', default='develop')
@click.option('--attempts', default=3, help='Number of times to retry job')
@click.option('--gpu', is_flag=True, help='Use CPU EC2 instances')
def batch_submit(command, repo, branch, attempts, gpu):
    if not is_branch_valid(repo, branch):
        exit(1)

    _batch_submit(command, repo=repo, branch=branch, attempts=attempts,
                  gpu=gpu)


if __name__ == '__main__':
    batch_submit()
