from typing import Callable
import unittest

from rastervision.core.utils.stac import setup_stac_io


class TestStac(unittest.TestCase):
    def assertNoError(self, fn: Callable, msg: str = ''):
        try:
            fn()
        except Exception:
            self.fail(msg)

    def test_setup_stac_io(self):
        self.assertNoError(setup_stac_io)


if __name__ == '__main__':
    unittest.main()
