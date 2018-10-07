# flake8: noqa
import logging

from rastervision.core import ConfigError
from rastervision.analyzer.api import *
from rastervision.augmentor.api import *
from rastervision.backend.api import *
from rastervision.command.api import *
from rastervision.data.api import *
from rastervision.evaluation.api import *
from rastervision.experiment.api import *
from rastervision.runner.api import *
from rastervision.task.api import *

from rastervision.predictor import Predictor

from rastervision.cli.main import main

from rastervision.registry import Registry

root_logger = logging.getLogger('rastervision')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

_registry = None


def _initialize():
    global _registry

    if not _registry:
        _registry = Registry()


_initialize()
