"""Helper functions to handle EC2 related operations with Boto3"""

import boto3
import logging


LOGGER = logging.getLogger(__file__)
LOGGER.addHandler(logging.StreamHandler())
LOGGER.setLevel(logging.INFO)


def _prune_ami(ec2, ami):
    """Actually deregister AMI and its associated snapshot"""
    LOGGER.info('Identified that [%s] is eligible to be pruned..',
                ami["ImageId"])

    ec2.deregister_image(ImageId=ami["ImageId"])
    map(lambda x: ec2.delete_snapshot(SnapshotId=x["Ebs"]["SnapshotId"]),
        ami["BlockDeviceMappings"])

    LOGGER.info('Deregistered [%s]', ami["ImageId"])


def prune(aws_profile, keep):
    """Filter owned AMIs by machine type, environment, and count

    Args:
      keep (int): number of images of this machine type to keep
      aws_profile (str): aws profile name to use for authentication
    """
    filters = [
        {"Name": "tag:Name", "Values": ["raster-vision-gpu"]}
    ]
    sess = boto3.Session(profile_name=aws_profile)
    ec2 = sess.client("ec2")
    images = ec2.describe_images(Owners=['self'], Filters=filters)["Images"]

    if len(images) > keep:
        map(lambda x: _prune_ami(ec2, x),
            list(sorted(images, key=lambda i: i["CreationDate"]))[0:len(images) - keep])  # NOQA
    else:
        LOGGER.info('No AMIs are eligible for pruning')
