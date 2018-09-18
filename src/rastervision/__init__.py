# flake8: noqa

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

from rastervision.cli.main import main

from rastervision.registry import Registry

_registry = None


def _initialize():
    global _registry

    if not _registry:
        _registry = Registry()


_initialize()
