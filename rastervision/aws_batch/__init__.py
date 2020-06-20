# flake8: noqa

import rastervision.pipeline
from rastervision.aws_batch.aws_batch_runner import *


def register_plugin(registry):
    registry.set_plugin_version('rastervision.aws_batch', 0)
    registry.set_plugin_aliases('rastervision.aws_batch',
                                ['rastervision2.aws_batch'])
    registry.add_runner(AWS_BATCH, AWSBatchRunner)
    registry.add_rv_config_schema(AWS_BATCH, [
        'gpu_job_queue', 'gpu_job_def', 'cpu_job_queue', 'cpu_job_def',
        'attempts'
    ])
