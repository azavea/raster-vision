# flake8: noqa
import logging
import importlib
import json

root_logger = logging.getLogger('rastervision2')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)

from rastervision2.pipeline.registry import Registry
_registry = Registry()

from rastervision2.pipeline.rv_config import RVConfig
_rv_config = RVConfig()

from rastervision2.pipeline.verbosity import Verbosity

from rastervision2.pipeline.runner import (InProcessRunner, INPROCESS)
from rastervision2.pipeline.filesystem import (HttpFileSystem, LocalFileSystem)

_registry.add_runner(INPROCESS, InProcessRunner)
_registry.add_filesystem(HttpFileSystem)
_registry.add_filesystem(LocalFileSystem)

# import so register_config decorators are called
import rastervision2.pipeline.pipeline_config

import importlib
import pkgutil
import rastervision2

# From https://packaging.python.org/guides/creating-and-discovering-plugins/#using-namespace-packages
def iter_namespace(ns_pkg):
    # Specifying the second argument (prefix) to iter_modules makes the
    # returned name an absolute name instead of a relative one. This allows
    # import_module to work without having to do additional modification to
    # the name.
    return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + ".")

discovered_plugins = {
    name: importlib.import_module(name)
    for finder, name, ispkg
    in iter_namespace(rastervision2)
}

for name, module in discovered_plugins.items():
    register_plugin = getattr(module, 'register_plugin', None)
    if register_plugin:
        register_plugin(_registry)
