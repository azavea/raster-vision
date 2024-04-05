import unittest

from rastervision.pipeline import (rv_config_ as rv_config)


class TestRVConfig(unittest.TestCase):
    def test_verbosity(self):
        rv_config.set_verbosity(0)
        self.assertEqual(rv_config.get_verbosity_cli_opt(), '')
        rv_config.set_verbosity(1)
        self.assertEqual(rv_config.get_verbosity_cli_opt(), '')
        rv_config.set_verbosity(2)
        self.assertEqual(rv_config.get_verbosity_cli_opt(), '-v')
        rv_config.set_verbosity(3)
        self.assertEqual(rv_config.get_verbosity_cli_opt(), '-vv')
        rv_config.set_verbosity(4)
        self.assertEqual(rv_config.get_verbosity_cli_opt(), '-vvv')


if __name__ == '__main__':
    unittest.main()
