# flake8: noqa


def register_plugin(registry):
    from rastervision.aws_sagemaker.aws_sagemaker_runner import (
        AWS_SAGEMAKER, AWSSageMakerRunner)
    registry.set_plugin_version('rastervision.aws_sagemaker', 0)
    registry.add_runner(AWS_SAGEMAKER, AWSSageMakerRunner)
    registry.add_rv_config_schema(AWS_SAGEMAKER, [
        'exec_role',
        'cpu_image',
        'cpu_inst_type',
        'gpu_image',
        'gpu_inst_type',
        'use_spot_instances',
    ])


import rastervision.pipeline
from rastervision.aws_sagemaker.aws_sagemaker_runner import *

__all__ = [
    'AWS_SAGEMAKER',
    AWSSageMakerRunner.__name__,
]
