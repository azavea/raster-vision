from typing import List, Union, Optional, Callable, Dict, TYPE_CHECKING
import inspect

from pydantic import BaseModel, create_model, Field, validator  # noqa
from typing_extensions import Literal

from rastervision.pipeline import registry
from rastervision.pipeline.file_system import str_to_file
from rastervision.pipeline import rv_config

if TYPE_CHECKING:
    from rastervision.pipeline.pipeline_config import PipelineConfig  # noqa


class ConfigError(ValueError):
    """Exception raised for invalid configuration."""
    pass


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

    @classmethod
    def get_field_summary(cls: type) -> str:
        """Returns class attributes PyDoc summarizing all Config fields."""
        summary = 'Attributes:\n'
        for _, field in cls.__fields__.items():
            if field.name != 'type_hint':
                desc = field.field_info.description or ''
                summary += '\t{} ({}): {}'.format(field.name,
                                                  field._type_display(), desc)
                if not field.required:
                    summary += '{} Defaults to {}.'.format(
                        '.' if desc and not desc.endswith('.') else '',
                        repr(field.default))
                summary += '\n'
        return summary

    def update(self):
        """Update any fields before validation.

        Subclasses should override this to provide complex default behavior, for
        example, setting default values as a function of the values of other
        fields. The arguments to this method will vary depending on the type of Config.
        """
        pass

    def build(self):
        """Build an instance of the corresponding type of object using this config.

        For example, BackendConfig will build a Backend object. The arguments to this
        method will vary depending on the type of Config.
        """
        pass

    def validate_config(self):
        """Validate fields that should be checked after update is called.

        This is to complement the builtin validation that Pydantic performs at the time
        of object construction.
        """
        pass

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
            field: name of field to validate
            valid_options: values that field is allowed to take

        Raises:
            ConfigError if field is invalid
        """
        val = getattr(self, field)
        if isinstance(val, list):
            for v in val:
                if v not in valid_options:
                    raise ConfigError('{} is not a valid option for {}'.format(
                        v, field))
        else:
            if val not in valid_options:
                raise ConfigError('{} is not a valid option for {}'.format(
                    val, field))


def save_pipeline_config(cfg: 'PipelineConfig', output_uri: str):
    """Save a PipelineConfig to JSON file.

    Inject rv_config and plugin_versions before saving.
    """
    cfg.rv_config = rv_config.get_config_dict(registry.rv_config_schema)
    cfg.plugin_versions = registry.plugin_versions
    cfg_json = cfg.json()
    str_to_file(cfg_json, output_uri)


def build_config(x: Union[dict, List[Union[dict, Config]], Config]
                 ) -> Union[Config, List[Config]]:  # noqa
    """Build a Config from various types of input.

    This is useful for deserializing from JSON. It implements polymorphic
    deserialization by using the `type_hint` in each dict to get the
    corresponding Config class from the registry.

    Args:
        x: some representation of Config(s)

    Returns:
        the corresponding Config(s)
    """
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = build_config(v)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
            # The following try/except logic has the following
            # motivation.  If one has an custom raster-vision pipeline
            # but wants to be able to run internal raster-vision
            # commands in processes or containers that do not have
            # that pipeline loaded in (e.g. in a "stock" raster-vision
            # container on AWS) then this code enables that.
            #
            # When a pipeline config is being loaded here, the first
            # attempt is to load it using the normal `type_hint` field
            # found within it.  If that succeeds, then everything
            # proceeds.
            #
            # If the the attempt fails but the pipeline config
            # contains a `fallback_type_hint`, then that is used,
            # instead.
            #
            # What that allows one to do is create a custom pipeline,
            # derived from a stock pipeline, and tell raster-vision to
            # treat it as a stock pipeline -- for purposes of the
            # present function -- if the custom pipeline has not been
            # registered.
            #
            # Please see https://github.com/azavea/raster-vision/pull/914
            try:
                # Try to use the given type hint
                config_cls = registry.get_config(type_hint)
            except Exception as e:
                # ... if that fails, try to downgrade to fallback type
                try:
                    type_hint = new_x.get('fallback_type_hint')
                    config_cls = registry.get_config(type_hint)
                    new_x['type_hint'] = type_hint
                    permitted_keys = config_cls().__dict__.keys()
                    current_keys = set(new_x.keys())
                    for k in current_keys:
                        if k not in permitted_keys:
                            del new_x[k]
                # ... if that fails, throw the original exception
                except Exception:
                    raise e
            new_x = config_cls(**new_x)
        return new_x
    elif isinstance(x, list):
        return [build_config(v) for v in x]
    else:
        return x


def _upgrade_config(x: Union[dict, List[dict]], plugin_versions: Dict[str, int]
                    ) -> Union[dict, List[dict]]:  # noqa
    """Upgrade serialized Config(s) to the latest version.

    Used to implement backward compatibility of Configs using upgraders stored
    in the registry.

    Args:
        x: serialized Config(s) which are potentially of a
            non-current version
        plugin_versions: dict mapping from plugin module name to the latest version

    Returns:
        the corresponding serialized Config(s) that have been upgraded to the
            current version
    """
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = _upgrade_config(v, plugin_versions)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
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

    Args:
        plugin_version: maps from plugin name to version

    This allows changing the names of plugins over time and maintaining backward
    compatibility of serialized PipelineConfigs.
    """
    new_plugin_versions = {}
    for alias, version in plugin_versions.items():
        plugin = registry.get_plugin_from_alias(alias)
        if plugin:
            new_plugin_versions[plugin] = version
        else:
            raise ConfigError(
                'The plugin_versions field contains an unrecognized '
                'plugin name: {}.'.format(alias))
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
        the corresponding serialized PipelineConfig(s) that have been upgraded to the
            current version
    """
    plugin_versions = config_dict.get('plugin_versions')
    plugin_versions = upgrade_plugin_versions(plugin_versions)
    if plugin_versions is None:
        raise ConfigError(
            'Configuration is missing plugin_version field so is not backward '
            'compatible.')
    return _upgrade_config(config_dict, plugin_versions)


def get_plugin(config_cls) -> str:
    """Infer the module path of the plugin where a Config class is defined.

    This only works correctly if the plugin is in a module under rastervision.
    """
    cls_module = inspect.getmodule(config_cls)
    return 'rastervision.' + cls_module.__name__.split('.')[1]


def register_config(type_hint: str,
                    plugin: Optional[str] = None,
                    upgrader: Optional[Callable] = None):
    """Class decorator used to register Config classes with registry.

    All Configs must be registered! Registering a Config does the following:
    1) associates Config classes with type_hint, plugin, and upgrader, which is
    necessary for polymorphic deserialization. See build_config() for more
    details.
    2) adds a constant `type_hint` field to the Config which is set to
    type_hint
    3) generates PyDocs based on Pydantic fields

    Args:
        type_hint: a type hint used to deserialize Configs. Needs to be unique
            across all registered Configs.
        plugin: the module path of the plugin where the Config is defined. This
            will be inferred if omitted.
        upgrader: a function of the form upgrade(config_dict, version) which returns the
            corresponding config dict of version = version + 1. This can be useful
            for maintaining backward compatibility by allowing old configs using an
            outdated schema to be upgraded to the current schema.
    """

    def _register_config(cls):
        new_cls = create_model(
            cls.__name__,
            type_hint=(Literal[type_hint], type_hint),
            __base__=cls)
        _plugin = plugin or get_plugin(cls)
        registry.add_config(type_hint, new_cls, _plugin, upgrader)

        new_cls.__doc__ = (cls.__doc__
                           or '') + '\n\n' + cls.get_field_summary()
        return new_cls

    return _register_config
