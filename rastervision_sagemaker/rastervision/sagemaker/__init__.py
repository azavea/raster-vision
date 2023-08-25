# flake8: noqa


def register_plugin(registry):
    from rastervision.sagemaker.sagemaker_runner import (SAGEMAKER,
                                                         SageMakerRunner)
    registry.set_plugin_version('rastervision.sagemaker', 0)
    registry.add_runner(SAGEMAKER, SageMakerRunner)
    registry.add_rv_config_schema(SAGEMAKER, [
        'exec_role',
        'cpu_image',
        'cpu_inst_type',
        'gpu_image',
        'gpu_inst_type',
    ])


import rastervision.pipeline
from rastervision.sagemaker.sagemaker_runner import *

__all__ = [
    'SAGEMAKER',
    SageMakerRunner.__name__,
    # submit_job.__name__,
]
