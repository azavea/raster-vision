#!/usr/bin/env python

import argparse
from os import environ

import boto3

s3_bucket = environ.get('S3_BUCKET')


def submit(branch_name, command_args, attempts=3, cpu=False):
    command = ['run_script.sh', branch_name]
    command.extend(command_args)

    client = boto3.client('batch')
    job_queue = 'raster-vision-cpu' if cpu else \
        'raster-vision-gpu'
    job_definition = 'raster-vision-cpu' if cpu else \
        'raster-vision-gpu'

    job_name = '-'.join(command_args) \
                  .replace('/', '-').replace('.', '-')
    job_name = 'batch_submit'

    job_id = client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            'command': command
        },
        retryStrategy={
            'attempts': attempts
        })['jobId']

    print(
        'Submitted job with jobName={} and jobId={}'.format(job_name, job_id))


def parse_args():
    description = """
        Submit a git branch and command to run on the GPU Docker container
        using AWS Batch.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('branch_name', help='Branch with code to run on AWS')
    parser.add_argument(
        'command', nargs='*',
        help='Space-delimited command to run in container on EC2.')
    parser.add_argument(
        '--attempts', type=int, default=3, help='Number of times to retry job')
    parser.add_argument(
        '--cpu', dest='cpu', action='store_true', help='Use CPU EC2 instances')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    submit(args.branch_name, args.command, attempts=args.attempts,
           cpu=args.cpu)
