import unittest

import rastervision as rv


class TestAnalyzeCommand(unittest.TestCase):
    def test_no_config_error(self):
        try:
            rv.command.AnalyzeCommandConfig.builder() \
                                           .with_task('') \
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
