# flake8: noqa
import logging
import importlib
import json
'''
root_logger = logging.getLogger('rastervision.v2')
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s: %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
sh.setFormatter(formatter)
root_logger.addHandler(sh)
'''

from rastervision.v2.core.registry import Registry
_registry = Registry()

from rastervision.v2.core.rv_config import RVConfig
_rv_config = RVConfig()

from rastervision.v2.core.verbosity import Verbosity

loaded_plugins = []


def load_builtins():
    from rastervision.v2.core.runner import (InProcessRunner, INPROCESS,
                                             AWSBatchRunner, AWS_BATCH)
    from rastervision.v2.core.filesystem import (HttpFileSystem, S3FileSystem,
                                                 LocalFileSystem)

    _registry.runners = {INPROCESS: InProcessRunner, AWS_BATCH: AWSBatchRunner}

    _registry.filesystems = [
        HttpFileSystem,
        S3FileSystem,
        # This is the catch-all case, ensure it's on the bottom of the search stack.
        LocalFileSystem
    ]


def load_conf_list(s):
    """Loads a list of items from the config.

    Lists should be comma separated.

    This takes into account that previous versions of Raster Vision
    allowed for a `[ "module" ]` like syntax, even though that didn't
    work for multi-value lists.
    """
    try:
        # A comma separated list of values will be transformed to
        # having a list-like string, with ' instead of ". Replacing
        # single quotes with double quotes lets us parse it as a JSON list.
        return json.loads(s.replace("'", '"'))
    except json.JSONDecodeError:
        return list(map(lambda x: x.strip(), s.split(',')))


def system_init(profile=None, verbosity=Verbosity.NORMAL):
    _rv_config.reset(profile=profile, verbosity=verbosity)

    plugin_config = _rv_config.get_subconfig('PLUGINS')
    if plugin_config:
        plugin_modules = load_conf_list(plugin_config('modules'))
        for pm in plugin_modules:
            if pm not in loaded_plugins:
                loaded_plugins.append(pm)
                pm = importlib.import_module(pm)
                register_plugin = getattr(pm, 'register_plugin', None)
                if register_plugin:
                    register_plugin(_registry)


load_builtins()
system_init()
'''
from rastervision.v2.core.config import *
from rastervision.v2.core.pipeline import *
from rastervision.v2.core.pipeline_config import *
'''
