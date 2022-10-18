# flake8: noqa


def register_plugin(registry):
    from rastervision.aws_batch.aws_batch_runner import (AWS_BATCH,
                                                         AWSBatchRunner)
    registry.set_plugin_version('rastervision.aws_batch', 0)
    registry.add_runner(AWS_BATCH, AWSBatchRunner)
    registry.add_rv_config_schema(AWS_BATCH, [
        'gpu_job_queue', 'gpu_job_def', 'cpu_job_queue', 'cpu_job_def',
        'attempts'
    ])


import rastervision.pipeline
from rastervision.aws_batch.aws_batch_runner import *

__all__ = [
    'AWS_BATCH',
    AWSBatchRunner.__name__,
    submit_job.__name__,
]
