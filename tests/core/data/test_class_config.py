from typing import Callable
import unittest

from rastervision.core.data.class_config import (
    ClassConfig, DEFAULT_NULL_CLASS_NAME, DEFAULT_NULL_CLASS_COLOR)
from rastervision.pipeline.config import ValidationError


class TestClassConfig(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_len_validation(self):
        args = dict(names=['a', 'b'], colors=['red', 'green'])
        self.assertNoError(lambda: ClassConfig(**args))

        args = dict(names=['a', 'b'], colors=['red'])
        self.assertRaises(ValidationError, lambda: ClassConfig(**args))

    def test_auto_colors_initialization(self):
        args = dict(names=['a', 'b'])
        self.assertNoError(lambda: ClassConfig(**args))

        cfg = ClassConfig(**args)
        self.assertEqual(len(cfg.names), len(cfg.colors))

    def test_null_class_validation(self):
        args = dict(names=['a', 'b'], null_class='a')
        self.assertNoError(lambda: ClassConfig(**args))

        args = dict(names=['a', 'b'], null_class='c')
        self.assertRaises(ValidationError, lambda: ClassConfig(**args))

        cfg = ClassConfig(
            names=['a', 'b', DEFAULT_NULL_CLASS_NAME], null_class=None)
        self.assertEqual(cfg.null_class, DEFAULT_NULL_CLASS_NAME)

        args = dict(names=['a', 'b', DEFAULT_NULL_CLASS_NAME], null_class='a')
        self.assertRaises(ValidationError, lambda: ClassConfig(**args))

    def test_ensure_null_class(self):
        cfg = ClassConfig(names=['a', 'b'])
        cfg.ensure_null_class()
        self.assertEqual(len(cfg.names), 3)
        self.assertEqual(len(cfg.colors), 3)
        self.assertEqual(cfg.names[-1], DEFAULT_NULL_CLASS_NAME)
        self.assertEqual(cfg.colors[-1], DEFAULT_NULL_CLASS_COLOR)

        # test idempotence
        cfg.ensure_null_class()
        cfg.ensure_null_class()
        cfg.ensure_null_class()
        self.assertEqual(len(cfg.names), 3)
        self.assertEqual(len(cfg.colors), 3)
        self.assertEqual(cfg.names[-1], DEFAULT_NULL_CLASS_NAME)
        self.assertEqual(cfg.colors[-1], DEFAULT_NULL_CLASS_COLOR)

        cfg = ClassConfig(names=['a', 'b'], null_class='a')
        cfg.ensure_null_class()
        self.assertEqual(len(cfg.names), 2)
        self.assertEqual(len(cfg.colors), 2)

        cfg = ClassConfig(
            names=['a', 'b'], colors=['red', DEFAULT_NULL_CLASS_COLOR])
        cfg.ensure_null_class()
        self.assertEqual(len(cfg.names), 3)
        self.assertEqual(len(cfg.colors), 3)
        self.assertEqual(cfg.names[-1], DEFAULT_NULL_CLASS_NAME)
        self.assertNotEqual(cfg.colors[-1], DEFAULT_NULL_CLASS_COLOR)

    def test_getters(self):
        cfg = ClassConfig(names=['a', 'b'], colors=['r', 'k'], null_class='a')
        self.assertEqual(cfg.get_class_id('a'), 0)
        self.assertEqual(cfg.get_class_id('b'), 1)
        self.assertRaises(ValueError, lambda: cfg.get_class_id('c'))

        self.assertEqual(cfg.get_name(0), 'a')
        self.assertEqual(cfg.get_name(1), 'b')
        self.assertRaises(IndexError, lambda: cfg.get_name(2))

        self.assertEqual(cfg.null_class_id, 0)

        color_to_class_id = cfg.get_color_to_class_id()
        self.assertEqual(len(color_to_class_id), 2)
        self.assertEqual(color_to_class_id['r'], 0)
        self.assertEqual(color_to_class_id['k'], 1)


if __name__ == '__main__':
    unittest.main()
