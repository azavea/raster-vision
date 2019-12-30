#!/usr/bin/env python3

import uuid

import click

from rastervision.rv_config import RVConfig


def _batch_submit(cmd,
                  debug=False,
                  profile=False,
                  attempts=5,
                  parent_job_ids=None,
                  num_array_jobs=None,
                  use_gpu=False):
    rv_config = RVConfig.get_instance()
    batch_config = rv_config.get_subconfig('AWS_BATCH')
    job_queue = batch_config('cpu_job_queue')
    job_def = batch_config('cpu_job_definition')
    if use_gpu:
        job_queue = batch_config('job_queue')
        job_def = batch_config('job_definition')

    import boto3
    client = boto3.client('batch')
    job_name = 'ffda-{}'.format(uuid.uuid4())

    cmd_list = cmd.split(' ')
    if debug:
        cmd_list = [
            'python', '-m', 'ptvsd', '--host', '0.0.0.0', '--port', '6006',
            '--wait', '-m'
        ] + cmd_list

    if profile:
        cmd_list = ['kernprof', '-v', '-l'] + cmd_list

    kwargs = {
        'jobName': job_name,
        'jobQueue': job_queue,
        'jobDefinition': job_def,
        'containerOverrides': {
            'command': cmd_list
        },
        'retryStrategy': {
            'attempts': attempts
        },
    }
    if parent_job_ids:
        kwargs['dependsOn'] = [{'jobId': id} for id in parent_job_ids]
    if num_array_jobs:
        kwargs['arrayProperties'] = {'size': num_array_jobs}

    job_id = client.submit_job(**kwargs)['jobId']
    msg = 'submitted job with jobName={} and jobId={}'.format(job_name, job_id)
    print(cmd_list)
    print(msg)
    return job_id


@click.command()
@click.argument('cmd')
@click.option('--debug', is_flag=True)
@click.option('--profile', is_flag=True)
@click.option('--attempts', default=5)
@click.option('--gpu', is_flag=True)
def batch_submit(cmd, debug, profile, attempts, gpu):
    return _batch_submit(cmd, debug, profile, attempts, use_gpu=gpu)


if __name__ == '__main__':
    batch_submit()
