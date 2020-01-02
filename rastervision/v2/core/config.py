from abc import ABC, abstractmethod

from pydantic import BaseModel, create_model
from typing_extensions import Literal

from rastervision.v2.core import _registry

class ConfigError(Exception):
    pass


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
            _registry.add_config_upgrader(type_hint, (version, upgraders))
        else:
            cls = create_model(
                cls.__name__,
                type_hint=(Literal[type_hint], type_hint),
                __base__=cls)

        _registry.add_config(type_hint, cls)
        return cls

    return _register_config


def build_config(x):
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = build_config(v)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
            config_cls = _registry.get_config(type_hint)
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
                curr_version, upgraders = _registry.get_config_upgrader(type_hint)
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


class Config(BaseModel):
    class Config:
        extra = 'forbid'

    def update(self, parent=None):
        pass

    def update_all(self, parent=None):
        self.update(parent=parent)

        for v in vars(self).values():
            if isinstance(v, Config):
                v.update(parent=self)
            elif isinstance(v, list):
                for _v in v:
                    if isinstance(_v, Config):
                        _v.update(parent=self)
