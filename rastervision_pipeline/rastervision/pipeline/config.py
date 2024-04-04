from typing import (TYPE_CHECKING, Callable, Dict, List, Literal, Optional,
                    Type, Union)
import inspect
import logging
import json

from pydantic import (  # noqa
    BaseModel, create_model, Field, root_validator, validate_model,
    ValidationError, validator)

from rastervision.pipeline import (registry_ as registry, rv_config_ as
                                   rv_config)
from rastervision.pipeline.file_system import (str_to_file, json_to_file,
                                               file_to_json)

if TYPE_CHECKING:
    from typing import Self
    from rastervision.pipeline.pipeline_config import PipelineConfig

log = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Exception raised for invalid configuration."""


class Config(BaseModel):
    """Base class that can be extended to provide custom configurations.

    This adds some extra methods to Pydantic BaseModel.
    See https://pydantic-docs.helpmanual.io/

    The general idea is that configuration schemas can be defined by
    subclassing this and adding class attributes with types and
    default values for each field. Configs can be defined hierarchically,
    ie. a Config can have fields which are of type Config.
    Validation, serialization, deserialization, and IDE support is
    provided automatically based on this schema.
    """

    # This is here to forbid instantiating Configs with fields that do not
    # exist in the schema, which helps avoid a command source of bugs.
    class Config:
        extra = 'forbid'
        validate_assignment = True

    def update(self, *args, **kwargs):
        """Update any fields before validation.

        Subclasses should override this to provide complex default behavior, for
        example, setting default values as a function of the values of other
        fields. The arguments to this method will vary depending on the type of Config.
        """

    def build(self):
        """Build an instance of the corresponding type of object using this config.

        For example, BackendConfig will build a Backend object. The arguments to this
        method will vary depending on the type of Config.
        """

    def validate_config(self):
        """Validate fields that should be checked after update is called.

        This is to complement the builtin validation that Pydantic performs at the time
        of object construction.
        """

    def revalidate(self):
        """Re-validate an instantiated Config.

        Runs all Pydantic validators plus self.validate_config().

        Adapted from:
        https://github.com/samuelcolvin/pydantic/issues/1864#issuecomment-679044432
        """
        *_, validation_error = validate_model(self.__class__, self.__dict__)
        if validation_error:
            raise validation_error
        self.validate_config()

    def recursive_validate_config(self):
        """Recursively validate hierarchies of Configs.

        This uses reflection to call validate_config on a hierarchy of Configs
        using a depth-first pre-order traversal.
        """
        class_hierarchy = type(self).mro()
        for klass in class_hierarchy:
            if issubclass(klass, Config):
                klass.validate_config(self)

        child_configs = [
            x for x in self.__dict__.values() if isinstance(x, Config)
        ]
        for c in child_configs:
            c.recursive_validate_config()

    def validate_list(self, field: str, valid_options: List[str]):
        """Validate a list field.

        Args:
            field (str): name of field to validate
            valid_options (List[str]): values that field is allowed to take

        Raises:
            ConfigError: if field is invalid
        """
        val = getattr(self, field)
        if isinstance(val, list):
            for v in val:
                if v not in valid_options:
                    raise ConfigError(f'{v} is not a valid option for {field}')
        else:
            if val not in valid_options:
                raise ConfigError(f'{val} is not a valid option for {field}')

    def to_file(self, uri: str, with_rv_metadata: bool = True) -> None:
        """Save a Config to a JSON file, optionally with RV metadata.

        Args:
            uri: URI to save to.
            with_rv_metadata: If True, inject Raster Vision metadata such as
                ``plugin_versions``, so that the config can be upgraded when
                loaded.
        """
        cfg_json = self.json()
        if with_rv_metadata:
            # self.dict() --> json_to_file() would be simpler but runs into
            # JSON serialization problems
            cfg_dict = json.loads(cfg_json)
            cfg_dict['plugin_versions'] = registry.plugin_versions
            cfg_json = json.dumps(cfg_dict)
        json_to_file(cfg_dict, uri)

    @classmethod
    def deserialize(cls, inp: 'str | dict | Config') -> 'Self':
        """Deserialize Config from a JSON file or dict, upgrading if possible.

        If ``inp`` is already a :class:`.Config`, it is returned as is.

        Args:
            inp: a URI to a JSON file or a dict.
        """
        if isinstance(inp, Config):
            return inp
        if isinstance(inp, dict):
            return cls.from_dict(inp)
        if isinstance(inp, str):
            return cls.from_file(inp)
        raise TypeError(f'Cannot deserialize Config from type: {type(inp)}.')

    @classmethod
    def from_file(cls, uri: str) -> 'Self':
        """Deserialize Config from a JSON file, upgrading if possible.

        Args:
            uri: URI to load from.
        """
        cfg_dict = load_config_dict(uri)
        cfg = cls.from_dict(cfg_dict)
        return cfg

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> 'Self':
        """Deserialize Config from a dict.

        Args:
            cfg_dict: Dict to deserialize.
        """
        cfg = build_config(cfg_dict)
        return cfg

    def __repr_args__(self):
        """Override to delete 'type_hint' field."""
        args = dict(super().__repr_args__())
        try:
            del args['type_hint']
        except KeyError:
            pass
        return args.items()


def save_pipeline_config(cfg: 'PipelineConfig', output_uri: str) -> None:
    """Save a PipelineConfig to JSON file.

    Inject rv_config and plugin_versions before saving.
    """
    cfg.rv_config = rv_config.get_config_dict(registry.rv_config_schema)
    cfg.plugin_versions = registry.plugin_versions
    cfg_json = cfg.json()
    str_to_file(cfg_json, output_uri)


def load_config_dict(uri: str) -> dict:
    """Load a serialized Config from a JSON file as a dict and upgrade it."""
    cfg_dict = file_to_json(uri)
    if 'plugin_versions' in cfg_dict:
        cfg_dict = upgrade_config(cfg_dict)
        cfg_dict.pop('plugin_versions', None)
    return cfg_dict


def build_config(x: Union[dict, List[Union[dict, Config]], Config]
                 ) -> Union[Config, List[Config]]:
    """Build a Config from various types of input.

    This is useful for deserializing from JSON. It implements polymorphic
    deserialization by using the `type_hint` in each dict to get the
    corresponding Config class from the registry.

    Args:
        x: some representation of Config(s)

    Returns:
        Config: the corresponding Config(s)
    """
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = build_config(v)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
            config_cls = registry.get_config(type_hint)
            new_x = config_cls(**new_x)
        return new_x
    elif isinstance(x, list):
        return [build_config(v) for v in x]
    else:
        return x


def _upgrade_config(x: Union[dict, List[dict]], plugin_versions: Dict[str, int]
                    ) -> Union[dict, List[dict]]:
    """Upgrade serialized Config(s) to the latest version.

    Used to implement backward compatibility of Configs using upgraders stored
    in the registry.

    Args:
        x: serialized Config(s) which are potentially of a non-current version
        plugin_versions: dict mapping from plugin module name to the latest version

    Returns:
        The corresponding serialized Config(s) that have been upgraded to the
        current version.
    """
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = _upgrade_config(v, plugin_versions)
        type_hint = new_x.get('type_hint')
        if type_hint is None:
            return new_x
        if type_hint in registry.renamed_type_hints:
            type_hint = registry.renamed_type_hints[type_hint]
            new_x['type_hint'] = type_hint
        type_hint_lineage = registry.get_type_hint_lineage(type_hint)
        for th in type_hint_lineage:
            plugin = registry.get_plugin(th)
            old_version = plugin_versions[plugin]
            curr_version = registry.get_plugin_version(plugin)
            upgrader = registry.get_upgrader(th)
            if upgrader:
                for version in range(old_version, curr_version):
                    new_x = upgrader(new_x, version)
        return new_x
    elif isinstance(x, list):
        return [_upgrade_config(v, plugin_versions) for v in x]
    else:
        return x


def upgrade_plugin_versions(plugin_versions: Dict[str, int]) -> Dict[str, int]:
    """Update the names of the plugins using the plugin aliases in the registry.

    This allows changing the names of plugins over time and maintaining backward
    compatibility of serialized PipelineConfigs.

    Args:
        plugin_version: maps from plugin name to version

    """
    new_plugin_versions = {}
    missing_plugins = []
    for alias, version in plugin_versions.items():
        plugin = registry.get_plugin_from_alias(alias)
        if plugin:
            new_plugin_versions[plugin] = version
        else:
            missing_plugins.append(alias)
    if len(missing_plugins) > 0:
        log.warning('There are plugins listed in the pipeline config that are '
                    f'not currently installed: {missing_plugins}')
    return new_plugin_versions


def upgrade_config(
        config_dict: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
    """Upgrade serialized Config(s) to the latest version.

    Used to implement backward compatibility of Configs using upgraders stored
    in the registry.

    Args:
        config_dict: serialized PipelineConfig(s) which are potentially of a
            non-current version

    Returns:
        The corresponding serialized PipelineConfig(s) that have been upgraded
        to the current version.
    """
    plugin_versions = config_dict.get('plugin_versions')
    plugin_versions = upgrade_plugin_versions(plugin_versions)
    if plugin_versions is None:
        raise ConfigError(
            'Configuration is missing plugin_version field so is not backward '
            'compatible.')
    return _upgrade_config(config_dict, plugin_versions)


def get_plugin(config_cls: Type) -> str:
    """Infer the module path of the plugin where a Config class is defined.

    This only works correctly if the plugin is in a module under rastervision.
    """
    cls_module = inspect.getmodule(config_cls)
    return 'rastervision.' + cls_module.__name__.split('.')[1]


def register_config(type_hint: str,
                    plugin: Optional[str] = None,
                    upgrader: Optional[Callable] = None) -> Callable:
    """Class decorator used to register Config classes with registry.

    All Configs must be registered! Registering a Config does the following:

    1.  Associates Config classes with type_hint, plugin, and upgrader, which
        is necessary for polymorphic deserialization. See build_config() for
        more details.
    2.  Adds a constant `type_hint` field to the Config which is set to
        type_hint.

    Args:
        type_hint (str): a type hint used to deserialize Configs. Must be
            unique across all registered Configs.
        plugin (Optional[str], optional): the module path of the plugin where
            the Config is defined. If None, will be inferred.
            Defauilts to None.
        upgrader (Optional[Callable], optional): a function of the form
            upgrade(config_dict, version) which returns the corresponding
            config dict of version = version + 1. This can be useful for
            maintaining backward compatibility by allowing old configs using an
            outdated schema to be upgraded to the current schema.
            Defaults to None.

    Returns:
        Callable: A function that returns a new class that is identical to the
        input Config with an additional ``type_hint`` field.
    """

    def _register_config(cls: Type):
        new_cls = create_model(
            cls.__name__,
            __base__=cls,
            __module__=cls.__module__,
            # add a new field called "type_hint" with type Literal[type_hint]
            # and default value type_hint to the config
            type_hint=(Literal[type_hint], type_hint),  # type: ignore
        )

        _plugin = plugin or get_plugin(cls)
        registry.add_config(type_hint, new_cls, _plugin, upgrader)

        # retain docstring after wrapping
        new_cls.__doc__ = cls.__doc__

        return new_cls

    return _register_config
