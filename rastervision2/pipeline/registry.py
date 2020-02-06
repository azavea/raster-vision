class RegistryError(Exception):
    pass


class Registry():
    def __init__(self):
        self.runners = {}
        self.filesystems = []
        self.configs = {}
        self.config_upgraders = {}
        self.rv_config_schema = {}

    def add_runner(self, runner_type, runner):
        if runner_type in self.runners:
            raise RegistryError(
                'There is already a {} runner in the registry.'.format(
                    runner_type))

        self.runners[runner_type] = runner

    def add_filesystem(self, filesystem):
        self.filesystems.append(filesystem)

    def get_runner(self, runner_type):
        runner = self.runners.get(runner_type)
        if runner:
            return runner
        else:
            raise RegistryError('{} is not a registered runner.'.format(runner_type))

    def get_file_system(self, uri: str, mode: str = 'r'):
        for fs in self.filesystems:
            if fs.matches_uri(uri, mode):
                return fs
        if mode == 'w':
            raise RegistryError('No matching filesystem to handle '
                                'writing to uri {}'.format(uri))
        else:
            raise RegistryError('No matching filesystem to handle '
                                'reading from uri {}'.format(uri))

    def get_config(self, type_hint):
        config = self.configs.get(type_hint)
        if config:
            return config
        else:
            raise RegistryError(
                '{} is not a registered config type hint.'.format(type_hint))

    def get_config_upgraders(self, type_hint):
        out = self.config_upgraders.get(type_hint)
        if out:
            return out
        else:
            raise RegistryError(
                '{} is not a registered config upgrader type hint.'.format(
                    type_hint))

    def add_config(self, type_hint, config_cls, version=0, upgraders=None):
        if type_hint in self.configs:
            raise RegistryError(
                'There is already a config registered for type_hint {}'.format(
                    type_hint))

        self.configs[type_hint] = config_cls

        if type_hint in self.config_upgraders:
            raise RegistryError(
                'There are already config upgraders registered for type_hint {}'.
                format(type_hint))
        self.config_upgraders[type_hint] = (version, upgraders)

    def add_rv_config_schema(self, config_section, config_fields):
        self.rv_config_schema[config_section] = config_fields

    def get_rv_config_schema(self):
        return self.rv_config_schema

    def load_plugins(self):
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
                register_plugin(self)

    def load_builtins(self):
        from rastervision2.pipeline.runner import (
            InProcessRunner, INPROCESS)
        from rastervision2.pipeline.filesystem import (
            HttpFileSystem, LocalFileSystem)

        self.add_runner(INPROCESS, InProcessRunner)
        self.add_filesystem(HttpFileSystem)
        self.add_filesystem(LocalFileSystem)

        # import so register_config decorators are called
        import rastervision2.pipeline.pipeline_config
