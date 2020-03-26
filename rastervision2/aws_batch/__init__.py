# flake8: noqa

import rastervision2.pipeline
from rastervision2.aws_batch.aws_batch_runner import *


def register_plugin(registry):
    registry.add_runner(AWS_BATCH, AWSBatchRunner)
    registry.add_rv_config_schema(AWS_BATCH, [
        'job_queue', 'job_definition', 'cpu_job_queue', 'cpu_job_definition',
        'attempts'
    ])
