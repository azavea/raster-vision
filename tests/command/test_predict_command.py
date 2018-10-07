import unittest

import rastervision as rv


class PredictCommand(unittest.TestCase):
    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.PredictCommandConfig.builder() \
                                           .with_backend('') \
                                           .with_scenes(['']) \
                                           .build()

    def test_missing_config_backend(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.PredictCommandConfig.builder() \
                                           .with_task('') \
                                           .with_scenes(['']) \
                                           .build()

    def test_missing_config_scenes(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.PredictCommandConfig.builder() \
                                           .with_task('') \
                                           .with_backend('') \
                                           .build()

    def test_no_config_error(self):
        try:
            rv.command.PredictCommandConfig.builder() \
                                           .with_task('') \
                                           .with_backend('') \
                                           .with_scenes(['']) \
                                           .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
