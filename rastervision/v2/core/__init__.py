# flake8: noqa
import logging
import importlib
import json

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

    import rastervision.v2.core.pipeline_config

    _registry.rv_config_schema['plugins'] = ['modules']
    _registry.rv_config_schema['aws_s3'] = ['requester_pays']
    _registry.rv_config_schema['aws_batch'] = [
        'job_queue', 'job_definition', 'cpu_job_queue', 'cpu_job_definition',
        'attempts']


def system_init(rv_config_dict=None, profile=None, verbosity=None):
    if verbosity is None:
        verbosity = Verbosity.NORMAL

    _rv_config.reset(
        config_overrides=rv_config_dict, profile=profile, verbosity=verbosity)
    plugin_modules = _rv_config.get_plugin_modules()

    for pm in plugin_modules:
        if pm not in loaded_plugins:
            loaded_plugins.append(pm)
            pm = importlib.import_module(pm)
            register_plugin = getattr(pm, 'register_plugin', None)
            if register_plugin:
                register_plugin(_registry)


load_builtins()
system_init()
