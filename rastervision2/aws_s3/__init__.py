# flake8: noqa

from rastervision2.aws_s3.s3_filesystem import (AWS_S3, S3FileSystem)


def register_plugin(registry):
    registry.add_filesystem(S3FileSystem)
    registry.add_rv_config_schema(AWS_S3, ['requester_pays'])
