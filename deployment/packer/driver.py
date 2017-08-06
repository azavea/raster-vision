"""Helper functions to handle AMI creation with packer"""

import boto3
import os
from os.path import dirname
import subprocess
import re
import logging

LOGGER = logging.getLogger(__file__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.INFO)


def get_recent_batch_ami(aws_profile):
    """Finds the most recent Deep Learning AMI"""
    filters = [
        {'Name': 'architecture', 'Values': ['x86_64']},
        {'Name': 'owner-id', 'Values': ['898082745236']},
        {'Name': 'virtualization-type', 'Values': ['hvm']},
    ]
    sess = boto3.Session(profile_name=aws_profile)
    ec2 = sess.client("ec2")
    images = filter(
        lambda x: re.search("Deep Learning AMI Amazon Linux .*", x['Name']),
        ec2.describe_images(Filters=filters)['Images'])
    images.sort(key=lambda x: x["CreationDate"])

    LOGGER.info("Found AMI: %s", images[-1]["ImageId"])

    return images[-1]["ImageId"]


def get_project_root():
    return dirname(dirname(dirname(os.path.realpath(__file__))))


def get_git_sha():
    """Function that executes Git to determine the current SHA"""
    git_command = ['git',
                   'rev-parse',
                   '--short',
                   'HEAD']

    src_dir = os.path.abspath(get_project_root())

    return subprocess.check_output(git_command, cwd=src_dir).rstrip()


def get_git_branch():
    """Function that executes Git to determine the current branch"""
    git_command = ['git',
                   'rev-parse',
                   '--abbrev-ref',
                   'HEAD']

    return subprocess.check_output(git_command).rstrip()


def run_packer(aws_profile, source_ami=None, aws_region="us-east-1"):
    """Function to run packer

    Args:
      aws_profile (str): aws profile name to use for authentication
      source_ami (str): AMI to use for Packer Builder
      aws_region (str): AWS region to user for Packer Builder
    """

    # Get AWS credentials based on profile
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile

    packer_template_path = os.path.join(get_project_root(),
                                        'deployment', 'packer', 'template.js')

    LOGGER.info('Creating GPU Workload AMI in %s region', aws_region)

    packer_command = [
        'packer', 'build',
        '-var', 'raster_vision_gpu_version={}'.format(get_git_sha()),
        '-var', 'aws_region={}'.format(aws_region),
        '-var', 'aws_gpu_ami={}'.format(source_ami or
                                        get_recent_batch_ami(aws_profile)),
        '-var', 'branch={}'.format(get_git_branch())]

    packer_command.append(packer_template_path)

    LOGGER.info('Running Packer Command: %s', ' '.join(packer_command))
    subprocess.check_call(packer_command, env=env)
