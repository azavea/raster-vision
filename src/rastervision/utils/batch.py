import uuid
import subprocess

import click
import boto3

from rastervision.utils.rv_config import get_rv_config


def is_branch_valid(repo, branch):
    ls_branch_command = [
        'git', 'ls-remote', '--heads', repo, branch]

    if not subprocess.run(ls_branch_command, stdout=subprocess.PIPE).stdout:
        print('Error: remote branch {} does not exist'.format(branch))
        return False
    return True


def _batch_submit(command, branch='develop', attempts=3, job_queue=None,
                  job_def=None, github_repo=None, profile=None,
                  parent_job_ids=[], array_size=None):
    """
        Submit a job to run on Batch.

        Args:
            repo: URI for Github repo with code to run
            branch: Branch with code to run
            command: Command in quotes to run
    """
    config = get_rv_config(batch_job_queue=job_queue, batch_job_def=job_def,
                           github_repo=github_repo, profile=profile)
    job_queue = config['batch_job_queue']
    job_def = config['batch_job_def']
    github_repo = config['github_repo']

    if not is_branch_valid(github_repo, branch):
        exit(1)

    full_command = ['run_rv', github_repo, branch]
    full_command.extend(command.split())

    client = boto3.client('batch')
    job_name = str(uuid.uuid4())
    depends_on = [{'jobId': job_id} for job_id in parent_job_ids]

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
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
@click.argument('command')
@click.option('--branch', default='develop')
@click.option('--attempts', default=3, help='Number of times to retry job')
@click.option('--job-queue', help='Batch job queue')
@click.option('--job-def', help='Batch job definition')
@click.option('--github-repo', help='Github repo with code to run')
@click.option('--profile', help='RV configuration profile to use')
def batch_submit(command, branch, attempts, job_queue, job_def,
                 github_repo, profile):
    _batch_submit(command, branch=branch, attempts=attempts,
                  job_queue=job_queue, job_def=job_def,
                  github_repo=github_repo, profile=profile)


if __name__ == '__main__':
    batch_submit()
