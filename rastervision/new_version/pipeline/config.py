from abc import ABC, abstractmethod

from pydantic import BaseModel, create_model
from typing_extensions import Literal

config_map = {}
upgrader_map = {}


def register_config(type_hint, version=0, upgraders=None):
    def _register_config(cls):
        if type_hint in config_map:
            raise ValueError(
                'There is already a config registered with type_hint="{}"'.
                format(type_hint))

        if version > 0:
            cls = create_model(
                cls.__name__,
                version=(Literal[version], version),
                type_hint=(Literal[type_hint], type_hint),
                __base__=cls)
            if upgraders is None or len(upgraders) != version:
                raise ValueError(
                    'If version > 0, must supply list of upgraders with length equal to version.'
                )
            upgrader_map[type_hint] = (version, upgraders)
        else:
            cls = create_model(
                cls.__name__,
                type_hint=(Literal[type_hint], type_hint),
                __base__=cls)

        config_map[type_hint] = cls
        return cls

    return _register_config


def build_config(x):
    if isinstance(x, dict):
        new_x = {}
        for k, v in x.items():
            new_x[k] = build_config(v)
        type_hint = new_x.get('type_hint')
        if type_hint is not None:
            config_cls = config_map.get(type_hint)
            if config_cls:
                new_x = config_cls(**new_x)
            else:
                raise ValueError(
                    'type_hint {} has not been registered'.format(type_hint))
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
                if type_hint in upgrader_map:
                    curr_version, upgraders = upgrader_map.get(type_hint)
                    for upgrader in upgraders[version:]:
                        new_x = upgrader.upgrade(new_x)
                    new_x['version'] = curr_version
                else:
                    raise ValueError('no upgraders for type_hint {}.')
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
