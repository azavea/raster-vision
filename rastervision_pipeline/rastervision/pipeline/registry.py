from typing import Iterable, List, Type, TYPE_CHECKING, Optional, Callable
import inspect
from click import Command

if TYPE_CHECKING:
    from rastervision.pipeline.runner import Runner  # noqa
    from rastervision.pipeline.file_system import FileSystem  # noqa
    from rastervision.pipeline.config import Upgrader, Config  # noqa


class RegistryError(Exception):
    """Exception raised for invalid use of registry."""


class Registry():
    """A registry for resources that are built-in or contributed by plugins."""

    def __init__(self):
        self.runners = {}
        self.file_systems = []
        self.configs = {}
        self.rv_config_schema = {}

        self.plugin_versions = {}
        self.alias_to_plugin = {}
        self.plugin_commands = []
        self.type_hint_to_lineage = {}
        self.type_hint_to_plugin = {}
        self.type_hint_to_upgrader = {}
        self.renamed_type_hints = {}

    def add_plugin_command(self, cmd: Command):
        """Add a click command contributed by a plugin."""
        self.plugin_commands.append(cmd)

    def get_plugin_commands(self) -> List[Command]:
        """Get the click commands contributed by plugins."""
        return self.plugin_commands

    def set_plugin_aliases(self, plugin: str, aliases: List[str]):
        self.alias_to_plugin[plugin] = plugin
        for alias in aliases:
            self.alias_to_plugin[alias] = plugin

    def get_plugin_from_alias(self, alias: str) -> Optional[str]:
        if alias in self.plugin_versions:
            return alias
        return self.alias_to_plugin.get(alias)

    def set_plugin_version(self, plugin: str, version: int):
        """Set the latest version of a plugin.

        Args:
            plugin: module path of plugin (eg. rastervision.core)
            version: a non-negative integer version number that should be incremented
                each time a config schema changes
        """
        self.plugin_versions[plugin] = version

    def register_renamed_type_hints(self, type_hint_old: str,
                                    type_hint_new: str):
        """Register renamed type_hints.

        Args:
            type_hint_old: Old type hint.
            type_hint_new: New type hint.
        """
        self.renamed_type_hints[type_hint_old] = type_hint_new

    def get_type_hint_lineage(self, type_hint: str) -> List[str]:
        """Get the lineage for a type hint.

        Returns:
            List of type hints of all Config classes in the subclass is-a
            "lineage" from Config to the class with type hint type_hint.
        """
        return self.type_hint_to_lineage[type_hint]

    def get_plugin_version(self, plugin: str) -> int:
        """Get latest version of plugin.

        Args:
            plugin: the module path of the plugin
        """
        return self.plugin_versions[plugin]

    def get_plugin(self, type_hint: str) -> str:
        """Get module path of plugin when Config class with type_hint is defined."""
        return self.type_hint_to_plugin[type_hint]

    def get_upgrader(self, type_hint: str) -> Optional[Callable]:
        """Get function that upgrades config dicts for type_hint."""
        return self.type_hint_to_upgrader.get(type_hint)

    def add_runner(self, runner_name: str, runner: Type['Runner']):
        """Add a Runner.

        Args:
            runner_name: the name of the runner that is passed to the CLI
            runner: the Runner class
        """
        if runner_name in self.runners:
            raise RegistryError(f'There is already a {runner_name} runner in '
                                'the registry.')

        self.runners[runner_name] = runner

    def get_runner(self, runner_name: str) -> Type['Runner']:  # noqa
        """Return a Runner class based on its name."""
        runner = self.runners.get(runner_name)
        if runner:
            return runner
        else:
            raise RegistryError(f'{runner_name} is not a registered runner.')

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
                                f'writing to uri {uri}')
        else:
            raise RegistryError('No matching file_system to handle '
                                f'reading from uri {uri}')

    def add_config(self,
                   type_hint: str,
                   config: Type['Config'],
                   plugin: str,
                   upgrader=None):
        """Add a Config.

        Args:
            type_hint: the type hint used for deserialization of dict to
                an instance of config
            config: Config class
        """
        if type_hint in self.configs:
            raise RegistryError('There is already a config registered for '
                                f'type_hint "{type_hint}".')

        self.configs[type_hint] = config
        self.type_hint_to_plugin[type_hint] = plugin
        if upgrader:
            self.type_hint_to_upgrader[type_hint] = upgrader

        self.update_config_info()

    def get_config(self, type_hint: str) -> Type['Config']:
        """Get a Config class associated with a type_hint."""
        config = self.configs.get(type_hint)
        if config:
            return config
        else:
            raise RegistryError(
                f'{type_hint} is not a registered config type hint. This may '
                'be because you forgot to use the register_config decorator, '
                'or forgot to import the module in the top-level __init__.py '
                'file for the plugin.')

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
        from rastervision.pipeline.runner import (InProcessRunner, INPROCESS,
                                                  LocalRunner, LOCAL)
        from rastervision.pipeline.file_system import (HttpFileSystem,
                                                       LocalFileSystem)

        self.add_runner(INPROCESS, InProcessRunner)
        self.add_runner(LOCAL, LocalRunner)
        self.add_file_system(HttpFileSystem)
        self.add_file_system(LocalFileSystem)

        # import so register_config decorators are called
        # TODO can we get rid of this now?
        import rastervision.pipeline.pipeline_config  # noqa
        self.set_plugin_version('rastervision.pipeline', 0)

    def update_config_info(self):
        config_class_to_type_hint = {}
        for type_hint, config_class in self.configs.items():
            config_class_to_type_hint[config_class] = type_hint

        for type_hint, config_class in self.configs.items():
            lineage = inspect.getmro(config_class)
            th_lineage = [
                config_class_to_type_hint[cc] for cc in lineage
                if config_class_to_type_hint.get(cc)
            ]
            th_lineage.reverse()
            self.type_hint_to_lineage[type_hint] = th_lineage

    def discover_plugins(self):
        """Discover all raster vision plugins."""
        import pkgutil
        import rastervision

        # From https://packaging.python.org/guides/creating-and-discovering-plugins/#using-namespace-packages  # noqa
        def iter_namespace(ns_pkg):
            # Specifying the second argument (prefix) to iter_modules makes the
            # returned name an absolute name instead of a relative one. This allows
            # import_module to work without having to do additional modification to
            # the name.
            return pkgutil.iter_modules(ns_pkg.__path__, ns_pkg.__name__ + '.')

        discovered_plugins = [
            name for _, name, _ in iter_namespace(rastervision)
        ]
        return discovered_plugins

    def load_plugins(self,
                     plugin_names: Optional[Iterable[str]] = None) -> None:
        """Load plugins and register their resources.

        Import each Python module within the rastervision namespace package
        and call the register_plugin function at its root (if it exists).
        """
        import importlib

        if plugin_names is None:
            plugin_names = self.discover_plugins()

        for name in plugin_names:
            module = importlib.import_module(name)
            register_plugin = getattr(module, 'register_plugin', None)
            if register_plugin:
                register_plugin(self)

        self.update_config_info()
