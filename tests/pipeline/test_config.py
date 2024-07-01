from typing import Callable, List
from os.path import join
import unittest

from rastervision.pipeline.file_system.utils import get_tmp_dir, json_to_file
from rastervision.pipeline.config import (
    Config, register_config, build_config, upgrade_config, ValidationError)
from rastervision.pipeline.pipeline_config import (PipelineConfig)
from rastervision.pipeline import registry_ as registry


def a_upgrader(cfg_dict, version):
    if version == 0:
        cfg_dict['x'] = cfg_dict['z']
        del cfg_dict['z']
    return cfg_dict


@register_config('a', plugin='rastervision.ab', upgrader=a_upgrader)
class AConfig(Config):
    x: str = 'x'


@register_config('asub1', plugin='rastervision.ab')
class ASub1Config(AConfig):
    y: str = 'y'


@register_config('asub2', plugin='rastervision.ab')
class ASub2Config(AConfig):
    y: str = 'y'


@register_config('b', plugin='rastervision.ab')
class BConfig(Config):
    x: str = 'x'


def c_upgrader(cfg_dict, version):
    if version == 0:
        cfg_dict['x'] = cfg_dict['y']
        del cfg_dict['y']
    return cfg_dict


@register_config('c', plugin='rastervision.c', upgrader=c_upgrader)
class CConfig(PipelineConfig):
    al: List[AConfig]
    bl: List[BConfig]
    a: AConfig
    b: BConfig
    x: str = 'x'


@register_config('c2', plugin='rastervision.c')
class CConfig2(Config):
    n: int
    abc: str


class TestConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def setUp(self):
        registry.set_plugin_version('rastervision.ab', 1)
        registry.set_plugin_version('rastervision.c', 1)
        registry.set_plugin_aliases('rastervision.c', ['rastervision2.c'])
        registry.update_config_info()
        self.plugin_versions = registry.plugin_versions

    def test_to_from(self):
        cfg = CConfig(
            al=[AConfig(), ASub1Config(),
                ASub2Config()],
            bl=[BConfig()],
            a=ASub1Config(),
            b=BConfig(),
            plugin_versions=self.plugin_versions,
            root_uri=None,
            rv_config=None)

        exp_dict = {
            'plugin_versions':
            self.plugin_versions,
            'root_uri':
            None,
            'rv_config':
            None,
            'type_hint':
            'c',
            'a': {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            },
            'al': [{
                'type_hint': 'a',
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
                'type_hint': 'b',
                'x': 'x'
            },
            'bl': [{
                'type_hint': 'b',
                'x': 'x'
            }],
            'x':
            'x'
        }

        self.assertDictEqual(cfg.dict(with_rv_metadata=False), exp_dict)
        self.assertEqual(build_config(exp_dict), cfg)

    def test_no_extras(self):
        with self.assertRaises(ValidationError):
            BConfig(zz='abc')

    def test_upgrade(self):
        plugin_versions_v0 = dict(self.plugin_versions)
        plugin_versions_v0['rastervision.ab'] = 0
        plugin_versions_v0['rastervision2.c'] = 0

        # after upgrading: the y field in the root should get converted to x, and
        # the z field in the instances of a should get convert to x.
        c_dict_v0 = {
            'plugin_versions':
            plugin_versions_v0,
            'root_uri':
            None,
            'rv_config':
            None,
            'type_hint':
            'c',
            'a': {
                'type_hint': 'asub1',
                'z': 'x',
                'y': 'y'
            },
            'al': [{
                'type_hint': 'a',
                'z': 'x'
            }, {
                'type_hint': 'asub1',
                'z': 'x',
                'y': 'y'
            }, {
                'type_hint': 'asub2',
                'z': 'x',
                'y': 'y'
            }],
            'b': {
                'type_hint': 'b',
                'x': 'x'
            },
            'bl': [{
                'type_hint': 'b',
                'x': 'x'
            }],
            'y':
            'x'
        }

        c_dict_v1 = {
            'plugin_versions':
            plugin_versions_v0,
            'root_uri':
            None,
            'rv_config':
            None,
            'type_hint':
            'c',
            'a': {
                'type_hint': 'asub1',
                'x': 'x',
                'y': 'y'
            },
            'al': [{
                'type_hint': 'a',
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
                'type_hint': 'b',
                'x': 'x'
            },
            'bl': [{
                'type_hint': 'b',
                'x': 'x'
            }],
            'x':
            'x'
        }

        upgraded_c_dict = upgrade_config(c_dict_v0)
        self.assertDictEqual(upgraded_c_dict, c_dict_v1)

    def test_deserialize(self):
        cfg = CConfig2(n=1, abc='abc')
        cfg_dict = cfg.dict()

        self.assertEqual(CConfig2.deserialize(cfg), cfg)

        self.assertNoError(lambda: CConfig2.deserialize(cfg_dict))
        cfg_deserialized = CConfig2.deserialize(cfg_dict)
        self.assertDictEqual(cfg_deserialized.dict(), cfg_dict)

        with get_tmp_dir() as tmp_dir:
            uri = join(tmp_dir, 'cfg.json')
            json_to_file(cfg_dict, uri)
            self.assertNoError(lambda: CConfig2.deserialize(uri))
            cfg_deserialized = CConfig2.deserialize(uri)
            self.assertDictEqual(cfg_deserialized.dict(), cfg_dict)

        self.assertRaises(TypeError, lambda: CConfig2.deserialize(0))


if __name__ == '__main__':
    unittest.main()
