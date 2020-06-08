# flake8: noqa

import rastervision2.pipeline
from rastervision2.aws_s3.s3_file_system import (AWS_S3, S3FileSystem)


def register_plugin(registry):
    registry.set_plugin_version('rastervision2.aws_s3', 0)
    registry.add_file_system(S3FileSystem)
    registry.add_rv_config_schema(AWS_S3, ['requester_pays'])
