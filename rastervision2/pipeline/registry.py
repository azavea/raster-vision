from typing import List, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from rastervision2.pipeline.runner import Runner  # noqa
    from rastervision2.pipeline.file_system import FileSystem  # noqa
    from rastervision2.pipeline.config import Upgrader, Config  # noqa


class RegistryError(Exception):
    """Exception raised for invalid use of registry."""
    pass


class Registry():
    """A registry for resources that are built-in or contributed by plugins."""

    def __init__(self):
        self.runners = {}
        self.file_systems = []
        self.configs = {}
        self.config_upgraders = {}
        self.rv_config_schema = {}

    def add_runner(self, runner_name: str, runner: Type['Runner']):
        """Add a Runner.

        Args:
            runner_name: the name of the runner that is passed to the CLI
            runner: the Runner class
        """
        if runner_name in self.runners:
            raise RegistryError(
                'There is already a {} runner in the registry.'.format(
                    runner_name))

        self.runners[runner_name] = runner

    def get_runner(self, runner_name: str) -> Type['Runner']:  # noqa
        """Return a Runner class based on its name."""
        runner = self.runners.get(runner_name)
        if runner:
            return runner
        else:
            raise RegistryError(
                '{} is not a registered runner.'.format(runner_name))

    def add_file_system(self, file_system: 'FileSystem'):
        """Add a FileSystem.

        Args:
            file_system: the FileSystem class to add
        """
        self.file_systems.append(file_system)

    def get_file_system(self, uri: str,
                        mode: str = 'r') -> Type['FileSystem']:  # noqa
        """Get a FileSystem used to handle the file type of a URI.

        Args:
            uri: a URI to be opened by a registered FileSystem
            mode: mode for opening file (eg. r or w)

        Returns:
            the first FileSystem class which can handle opening the uri
        """
        for fs in self.file_systems:
            if fs.matches_uri(uri, mode):
                return fs
        if mode == 'w':
            raise RegistryError('No matching file_system to handle '
                                'writing to uri {}'.format(uri))
        else:
            raise RegistryError('No matching file_system to handle '
                                'reading from uri {}'.format(uri))

    def add_config(self,
                   type_hint: str,
                   config: Type['Config'],
                   version: int = 0,
                   upgraders: List['Upgrader'] = None):
        """Add a Config.

        Args:
            type_hint: the type hint used for deserialization of dict to
                an instance of config
            config: Config class
            version: the current version of the Config
            upgraders: a sequence of Upgraders that go from version 0 to
                version
        """
        if type_hint in self.configs:
            raise RegistryError(
                'There is already a config registered for type_hint {}'.format(
                    type_hint))

        self.configs[type_hint] = config

        if type_hint in self.config_upgraders:
            raise RegistryError(
                'There are already config upgraders registered for type_hint {}'.
                format(type_hint))
        self.config_upgraders[type_hint] = (version, upgraders)

    def get_config(self, type_hint: str) -> Type['Config']:
        """Get a Config class associated with a type_hint."""
        config = self.configs.get(type_hint)
        if config:
            return config
        else:
            raise RegistryError(
                ('{} is not a registered config type hint.'
                 'This may be because you forgot to use the register_config decorator, '
                 'or forgot to import the module in the top-level __init__.py file for '
                 'the plugin.').format(type_hint))

    def get_config_upgraders(self, type_hint: str) -> List['Upgrader']:  # noqa
        """Get config upgraders associated with type_hint."""
        out = self.config_upgraders.get(type_hint)
        if out:
            return out
        else:
            raise RegistryError(
                '{} is not a registered config upgrader type hint.'.format(
                    type_hint))

    def add_rv_config_schema(self, config_section: str,
                             config_fields: List[str]):
        """Add section of schema used by RVConfig.

        Args:
            config_section: name of section
            config_fields: list of field names within section
        """
        self.rv_config_schema[config_section] = config_fields

    def get_rv_config_schema(self):
        """Return RVConfig schema."""
        return self.rv_config_schema

    def load_builtins(self):
        """Add all builtin resources."""
        from rastervision2.pipeline.runner import (
            InProcessRunner, INPROCESS, LocalRunner, LOCAL)
        from rastervision2.pipeline.file_system import (HttpFileSystem,
                                                        LocalFileSystem)

        self.add_runner(INPROCESS, InProcessRunner)
        self.add_runner(LOCAL, LocalRunner)
        self.add_file_system(HttpFileSystem)
        self.add_file_system(LocalFileSystem)

        # import so register_config decorators are called
        # TODO can we get rid of this now?
        import rastervision2.pipeline.pipeline_config  # noqa

    def load_plugins(self):
        """Discover all plugins and register their resources.

        Import each Python module within the rastervision2 namespace package
        and call the register_plugin function at its root (if it exists).
        """
        import importlib
        import pkgutil
        import rastervision2

        # From https://packaging.python.org/guides/creating-and-discovering-plugins/#using-namespace-packages  # noqa
        def iter_namespace(ns_pkg):
            # Specifying the second argument (prefix) to iter_modules makes the
            # returned name an absolute name instead of a relative one. This allows
            # import_module to work without having to do additional modification to
            # the name.
            return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + '.')

        discovered_plugins = {
            name: importlib.import_module(name)
            for finder, name, ispkg in iter_namespace(rastervision2)
        }

        for name, module in discovered_plugins.items():
            register_plugin = getattr(module, 'register_plugin', None)
            if register_plugin:
                register_plugin(self)
