from abc import ABC, abstractmethod

from pydantic import BaseModel, create_model
from typing_extensions import Literal

from rastervision2.pipeline import registry


def register_config(type_hint, version=0, upgraders=None):
    def _register_config(cls):
        if version > 0:
            cls = create_model(
                cls.__name__,
                version=(Literal[version], version),
                type_hint=(Literal[type_hint], type_hint),
                __base__=cls)
            if upgraders is None or len(upgraders) != version:
                raise ValueError(
                    'If version > 0, must supply list of upgraders with length'
                    ' equal to version.')
        else:
            cls = create_model(
                cls.__name__,
                type_hint=(Literal[type_hint], type_hint),
                __base__=cls)
        registry.add_config(
            type_hint, cls, version=version, upgraders=upgraders)
        return cls

    return _register_config


def build_config(x):
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


def upgrade_config(x):
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = upgrade_config(v)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
            version = new_x.get('version')
            if version is not None:
                curr_version, upgraders = registry.get_config_upgraders(
                    type_hint)
                for upgrader in upgraders[version:]:
                    new_x = upgrader.upgrade(new_x)
                new_x['version'] = curr_version
        return new_x
    elif isinstance(x, list):
        return [upgrade_config(v) for v in x]
    else:
        return x


class Upgrader(ABC):
    @abstractmethod
    def upgrade(self, cfg_dict):
        pass


class ConfigError(ValueError):
    pass


class Config(BaseModel):
    class Config:
        extra = 'forbid'

    def update(self):
        pass

    def validate_config(self):
        pass

    def recursive_validate_config(self):
        class_hierarchy = type(self).mro()
        for klass in class_hierarchy:
            if issubclass(klass, Config):
                klass.validate_config(self)

        child_configs = [
            x for x in self.__dict__.values() if isinstance(x, Config)
        ]
        for c in child_configs:
            c.recursive_validate_config()

    def validate_list(self, field, valid_options):
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

    def validate_nonneg(self, field):
        if getattr(self, field) < 0:
            raise ConfigError('{} cannot be negative'.format(field))
