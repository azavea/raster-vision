#!/usr/bin/env python
"""Commands for Creating AWS Batch GPU Worker AMIs"""

import argparse

from packer.driver import run_packer
from ec2.amis import prune


def create_ami(aws_profile, aws_region, source_ami=None, **kwargs):
    run_packer(aws_profile, source_ami, aws_region=aws_region)


def prune_ami(aws_profile, keep, **kwargs):
    prune(aws_profile, keep)


def main():
    """Parse args and run desired commands"""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--aws-profile', default='raster-vision',
                               help='AWS profile to use for launching the'
                                    'builder EC2 instance')
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='AMI Commands')
    build_amis = subparsers.add_parser('build-amis',
                                       help='Build batch AMIs',
                                       parents=[common_parser])
    build_amis.add_argument('--aws-region', default='us-east-1',
                            help='AWS region to use for launching an EC2 '
                                 'instance')

    build_amis.add_argument('--source-ami-id', default=None,
                            help='ID of the base ami to use for launching the '
                                 'EC2 instance.')
    build_amis.set_defaults(func=create_ami)

    prune_amis = subparsers.add_parser('prune-amis',
                                       help='Prune old AMIs',
                                       parents=[common_parser])
    prune_amis.add_argument('--keep', default=10, type=int,
                            help='Number of old AMIs to keep')
    prune_amis.set_defaults(func=prune_ami)

    args = parser.parse_args()
    args.func(**vars(args))


if __name__ == '__main__':
    main()
