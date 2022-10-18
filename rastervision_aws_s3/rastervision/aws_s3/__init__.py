# flake8: noqa


def register_plugin(registry):
    from rastervision.aws_s3.s3_file_system import (AWS_S3, S3FileSystem)
    registry.set_plugin_version('rastervision.aws_s3', 0)
    registry.add_file_system(S3FileSystem)
    registry.add_rv_config_schema(AWS_S3, ['requester_pays'])


import rastervision.pipeline
from rastervision.aws_s3.s3_file_system import *

__all__ = [
    'AWS_S3',
    S3FileSystem.__name__,
]
