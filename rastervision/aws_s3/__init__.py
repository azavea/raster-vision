# flake8: noqa

import rastervision.pipeline
from rastervision.aws_s3.s3_file_system import (AWS_S3, S3FileSystem)


def register_plugin(registry):
    registry.set_plugin_version('rastervision.aws_s3', 0)
    registry.set_plugin_aliases('rastervision.aws_s3',
                                ['rastervision2.aws_s3'])
    registry.add_file_system(S3FileSystem)
    registry.add_rv_config_schema(AWS_S3, ['requester_pays'])
