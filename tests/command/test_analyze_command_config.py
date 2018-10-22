import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig


class TestAnalyzeCommand(unittest.TestCase):
    def test_command_create(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            cmd = rv.command.AnalyzeCommandConfig.builder() \
                                                 .with_task('') \
                                                 .with_root_uri(tmp_dir) \
                                                 .with_scenes('') \
                                                 .with_analyzers('') \
                                                 .build() \
                                                 .create_command()
            self.assertTrue(cmd, rv.command.AnalyzeCommand)

    def test_no_config_error(self):
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.AnalyzeCommandConfig.builder() \
                                               .with_task('') \
                                               .with_root_uri(tmp_dir) \
                                               .with_scenes('') \
                                               .with_analyzers('') \
                                               .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')

    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.AnalyzeCommandConfig.builder() \
                                           .with_scenes('') \
                                           .with_analyzers('') \
                                           .build()

    def test_missing_config_scenes(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.AnalyzeCommandConfig.builder() \
                                           .with_task('') \
                                           .with_analyzers('') \
                                           .build()

    def test_missing_config_analyzers(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.AnalyzeCommandConfig.builder() \
                                           .with_task('') \
                                           .with_scenes('') \
                                           .build()


if __name__ == '__main__':
    unittest.main()
