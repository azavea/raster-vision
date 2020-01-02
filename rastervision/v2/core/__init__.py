# flake8: noqa
import logging

root_logger = logging.getLogger('rastervision.v2')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

from rastervision.v2.core.rv_config import RVConfig
_rv_config = RVConfig()

from rastervision.v2.core.registry import Registry
_registry = Registry()

from rastervision.v2.core.runner import (
    InProcessRunner, INPROCESS, AWSBatchRunner, AWS_BATCH)
from rastervision.v2.core.filesystem import (
    HttpFileSystem, S3FileSystem, LocalFileSystem)

_registry.runners = {
    INPROCESS: InProcessRunner,
    AWS_BATCH: AWSBatchRunner
}

_registry.filesystems = [
    HttpFileSystem,
    S3FileSystem,
    # This is the catch-all case, ensure it's on the bottom of the search stack.
    LocalFileSystem
]

import rastervision.v2.core.pipeline_config

'''
import rastervision.v2.learner.learner_config
import rastervision.v2.learner.regression_config
import rastervision.v2.learner.classification_config
'''
