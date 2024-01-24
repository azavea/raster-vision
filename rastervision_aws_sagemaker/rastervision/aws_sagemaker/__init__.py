# flake8: noqa


def register_plugin(registry):
    from rastervision.aws_sagemaker.aws_sagemaker_runner import (
        AWS_SAGEMAKER, AWSSageMakerRunner)
    registry.set_plugin_version('rastervision.aws_sagemaker', 0)
    registry.add_runner(AWS_SAGEMAKER, AWSSageMakerRunner)
    registry.add_rv_config_schema(AWS_SAGEMAKER, [
        'role',
        'cpu_image',
        'cpu_instance_type',
        'gpu_image',
        'gpu_instance_type',
        'train_image',
        'train_instance_type',
        'train_instance_count',
        'use_spot_instances',
        'spot_instance_max_wait_time',
        'max_run_time',
    ])


import rastervision.pipeline
from rastervision.aws_sagemaker.aws_sagemaker_runner import *

__all__ = [
    'AWS_SAGEMAKER',
    AWSSageMakerRunner.__name__,
]
