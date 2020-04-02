# flake8: noqa

import rastervision2.pipeline
from rastervision2.aws_batch.aws_batch_runner import *


def register_plugin(registry):
    registry.add_runner(AWS_BATCH, AWSBatchRunner)
    registry.add_rv_config_schema(AWS_BATCH, [
        'gpu_job_queue', 'gpu_job_def', 'cpu_job_queue', 'cpu_job_def',
        'attempts'
    ])
