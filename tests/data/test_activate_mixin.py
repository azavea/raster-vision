import unittest

from rastervision.data import (ActivateMixin, ActivationError)


class TestActivateMixin(unittest.TestCase):
    class Foo(ActivateMixin):
        def __init__(self):
            self.activated = False

        def _activate(self):
            self.activated = True

        def _deactivate(self):
            self.activated = False

    class Bar(ActivateMixin):
        def __init__(self):
            self.activated = False
            self.foo = TestActivateMixin.Foo()

        def _activate(self):
            self.activated = True

        def _deactivate(self):
            self.activated = False

        def _subcomponents_to_activate(self):
            return [self.foo]

    def test_activates_and_deactivates(self):
        foo = TestActivateMixin.Foo()
        self.assertFalse(foo.activated)
        with foo.activate():
            self.assertTrue(foo.activated)
        self.assertFalse(foo.activated)

    def test_activated_and_deactivates_subcomponents(self):
        bar = TestActivateMixin.Bar()
        self.assertFalse(bar.activated)
        self.assertFalse(bar.foo.activated)
        with bar.activate():
            self.assertTrue(bar.activated)
            self.assertTrue(bar.foo.activated)
        self.assertFalse(bar.activated)
        self.assertFalse(bar.foo.activated)

    def test_no_activate_twice(self):
        bar = TestActivateMixin.Bar()
        with self.assertRaises(ActivationError):
            with bar.activate():
                with bar.activate():
                    pass
        self.assertFalse(bar.activated)
