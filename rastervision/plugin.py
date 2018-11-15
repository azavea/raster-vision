import os
import json
import importlib

from pluginbase import PluginBase

import rastervision as rv
from rastervision.protos.plugin_pb2 import PluginConfig as PluginConfigMsg
from rastervision.utils.files import download_if_needed


class PluginError(Exception):
    pass


class PluginRegistry:
    @staticmethod
    def get_instance():
        return rv._registry._get_plugin_registry()

    def __init__(self, plugin_config, rv_home):
        """Initializes this plugin registry.

        A plugin registry is passed to plugins in a call
        to their "register_plugin" method.

        Args:
           plugin_config - the everett ConfigManager for the plugin
                           section of the application configuration.
        """
        self.plugin_root_dir = os.path.join(rv_home, 'plugins')
        self.config_builders = {}
        self.default_raster_sources = []
        self.default_vector_sources = []
        self.default_label_sources = []
        self.default_label_stores = []
        self.default_evaluators = []
        self.experiment_runners = {}
        self.filesystems = []

        plugin_files = json.loads(plugin_config('files', default='[]'))
        self._load_from_files(plugin_files)
        self.plugin_files = plugin_files

        plugin_modules = json.loads(plugin_config('modules', default='[]'))
        self._load_from_modules(plugin_modules)
        self.plugin_modules = plugin_modules

    def _load_plugin(self, plugin, identifier):
        # Check the plugin is valid
        if not hasattr(plugin, 'register_plugin'):
            raise PluginError('Plugin at {} does not have '
                              '"register_plugin" method.'.format(identifier))

        register_method = getattr(plugin, 'register_plugin')
        if not callable(register_method):
            raise PluginError('Plugin at {} has a '
                              '"register_plugin" attribute, '
                              'but it is not callable'.format(identifier))

        # TODO: Log loading plugin.
        register_method(self)

    def _load_from_files(self, plugin_paths):
        if not plugin_paths:
            return

        self.plugin_sources = []

        plugin_base = PluginBase(package='rastervision.plugins')
        for uri in plugin_paths:
            plugin_name = os.path.splitext(os.path.basename(uri))[0]
            plugin_path = os.path.join(self.plugin_root_dir, plugin_name)
            fs = rv._registry.get_file_system(uri, search_plugins=False)
            local_path = download_if_needed(uri, plugin_path, fs=fs)
            local_dir = os.path.dirname(local_path)

            plugin_source = plugin_base.make_plugin_source(
                searchpath=[local_dir])

            # We're required to hang onto the source
            # to keep it from getting GC'd.
            self.plugin_sources.append(plugin_source)

            self._load_plugin(plugin_source.load_plugin(plugin_name), uri)

    def _load_from_modules(self, plugin_modules):
        if not plugin_modules:
            return

        for module in plugin_modules:
            plugin = importlib.import_module(module)
            self._load_plugin(plugin, module)

    def add_plugins_from_proto(self, plugin_msg):
        new_plugin_files = list(
            set(plugin_msg.plugin_uris) - set(self.plugin_files))
        self._load_from_files(new_plugin_files)
        self.plugin_files.extend(new_plugin_files)

        new_plugin_modules = list(
            set(plugin_msg.plugin_modules) - set(self.plugin_modules))
        self._load_from_modules(new_plugin_modules)
        self.plugin_modules.extend(new_plugin_modules)

    def to_proto(self):
        """Returns a protobuf message that records the
        plugin sources for plugins that are currently loaded
        in the registry.
        """
        return PluginConfigMsg(
            plugin_uris=self.plugin_files, plugin_modules=self.plugin_modules)

    def register_config_builder(self, group, key, builder_class):
        """Registers a ConfigBuilder as a plugin.

        Args:
           group - The Config group, e.g. rv.BACKEND, rv.TASK.
           key - The key used for this plugin. This will be used to
                 construct the builder in a ".builder(key)" call.
           builder_class - The subclass of ConfigBuilder that builds
                           the Config for this plugin.
        """
        if (group, key) in self.config_builders:
            raise PluginError('ConfigBuilder already registered for group '
                              '{} and key {}'.format(group, key))
        self.config_builders[(group, key)] = builder_class

    def register_default_raster_source(self, provider_class):
        """Registers a RasterSourceDefaultProvider for use as a plugin."""

        self.default_raster_sources.append(provider_class)

    def register_default_vector_source(self, provider_class):
        """Registers a VectorSourceDefaultProvider for use as a plugin."""
        self.default_vector_sources.append(provider_class)

    def register_default_label_source(self, provider_class):
        """Registers a LabelSourceDefaultProvider for use as a plugin."""
        self.default_label_sources.append(provider_class)

    def register_default_label_store(self, provider_class):
        """Registers a LabelStoreDefaultProvider for use as a plugin."""
        self.default_label_stores.append(provider_class)

    def register_default_evaluator(self, provider_class):
        """Registers an EvaluatorDefaultProvider for use as a plugin."""
        self.default_evaluators.append(provider_class)

    def register_experiment_runner(self, runner_key, runner_class):
        """Registers an ExperimentRunner as a plugin.

        Args:
           runner_key - The key used to reference this plugin runner.
                        This is a string that will match the command line
                        argument used to reference this runner; e.g. if the
                        key is "FOO_RUNNER", then users can use the runner
                        by issuing a "rastervision run  foo_runner ..." command.
           runner_class - The class of the ExperimentRunner plugin.
        """
        if runner_key in self.experiment_runners:
            raise PluginError('ExperimentRunner already registered for '
                              'key {}'.format(runner_key))
        self.experiment_runners[runner_key] = runner_class

    def register_filesystem(self, filesystem_class):
        """Registers a FileSystem as a plugin."""
        self.filesystems.append(filesystem_class)
