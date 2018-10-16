import unittest

from rastervision.cli import Verbosity


class TestVerbosity(unittest.TestCase):
    def test_verbosity(self):
        self.assertTrue(type(Verbosity.get()) == int)


if __name__ == '__main__':
    unittest.main()
