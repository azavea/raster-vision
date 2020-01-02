from typing import List
import unittest
import copy

from pydantic.error_wrappers import ValidationError

from rastervision.v2.core.config import (
    Config, register_config, build_config, upgrade_config, Upgrader)


class AConfig(Config):
    x: str = 'x'


@register_config('asub1')
class ASub1Config(AConfig):
    y: str = 'y'


@register_config('asub2')
class ASub2Config(AConfig):
    y: str = 'y'


class BConfig(Config):
    x: str = 'x'


class UpgradeC1(Upgrader):
    def upgrade(self, cfg_dict):
        cfg_dict = copy.deepcopy(cfg_dict)
        cfg_dict['x'] = cfg_dict['y']
        del cfg_dict['y']
        return cfg_dict


@register_config('c', version=1, upgraders=[UpgradeC1()])
class CConfig(Config):
    al: List[AConfig]
    bl: List[BConfig]
    a: AConfig
    b: BConfig
    x: str = 'x'


class TestConfig(unittest.TestCase):
    def test_to_from(self):
        cfg = CConfig(
            al=[AConfig(), ASub1Config(),
                ASub2Config()],
            bl=[BConfig()],
            a=ASub1Config(),
            b=BConfig())

        exp_dict = {
            'type_hint':
            'c',
            'version':
            1,
            'a': {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            },
            'al': [{
                'x': 'x'
            }, {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            }, {
                'type_hint': 'asub2',
                'x': 'x',
                'y': 'y'
            }],
            'b': {
                'x': 'x'
            },
            'bl': [{
                'x': 'x'
            }],
            'x':
            'x'
        }

        self.assertDictEqual(cfg.dict(), exp_dict)
        self.assertEqual(build_config(exp_dict), cfg)

    def test_no_extras(self):
        with self.assertRaises(ValidationError):
            BConfig(zz='abc')

    def test_upgrade(self):
        c_dict_v0 = {
            'type_hint':
            'c',
            'version':
            0,
            'a': {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            },
            'al': [{
                'x': 'x'
            }, {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            }, {
                'type_hint': 'asub2',
                'x': 'x',
                'y': 'y'
            }],
            'b': {
                'x': 'x'
            },
            'bl': [{
                'x': 'x'
            }],
            'y':
            'x'
        }

        c_dict_v1 = {
            'type_hint':
            'c',
            'version':
            1,
            'a': {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            },
            'al': [{
                'x': 'x'
            }, {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            }, {
                'type_hint': 'asub2',
                'x': 'x',
                'y': 'y'
            }],
            'b': {
                'x': 'x'
            },
            'bl': [{
                'x': 'x'
            }],
            'x':
            'x'
        }
        upgraded_c_dict = upgrade_config(c_dict_v0)
        self.assertDictEqual(upgraded_c_dict, c_dict_v1)


if __name__ == '__main__':
    unittest.main()
