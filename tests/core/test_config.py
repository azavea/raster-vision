from copy import deepcopy
import unittest
from rastervision.core.config import ConfigBuilder


class DummyConfig(object):
    def __init__(self, foo: str):
        self.foo = foo


class DummyConfigBuilder(ConfigBuilder):
    config_class = DummyConfig
    config = {'foo': 'bar'}

    def __init__(self):
        return

    def with_foo(self, foo):
        b = deepcopy(self)
        b.foo = foo
        return b

    def from_proto(self, msg):
        return self.with_foo(msg.foo)


class TestConfig(unittest.TestCase):
    def test_build_with_annotations(self):
        self.assertTrue(DummyConfigBuilder().build().foo == 'bar')
