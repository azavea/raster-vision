# flake8: noqa
import logging
import json

from rastervision2.pipeline.rv_config import RVConfig
from rastervision2.pipeline.registry import Registry
from rastervision2.pipeline.verbosity import Verbosity

root_logger = logging.getLogger('rastervision2')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

rv_config = RVConfig()
registry = Registry()
registry.load_plugins()
registry.load_builtins()
