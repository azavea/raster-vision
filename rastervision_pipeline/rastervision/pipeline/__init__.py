# flake8: noqa
import logging
import json

# torch needs to be imported before anything else in RV or we will get a
# segmentation fault when calling model.to('cuda'). This is very weird, and not
# a great solution, but the show must go on.
import torch

from rastervision.pipeline.rv_config import RVConfig
from rastervision.pipeline.registry import Registry
from rastervision.pipeline.verbosity import Verbosity

root_logger = logging.getLogger('rastervision')
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
