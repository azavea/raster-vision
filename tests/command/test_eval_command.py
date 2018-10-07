import unittest

import rastervision as rv


class TestEvalCommand(unittest.TestCase):
    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.EvalCommandConfig.builder() \
                                        .with_scenes('') \
                                        .with_evaluators('') \
                                        .build()

    def test_missing_config_scenes(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.EvalCommandConfig.builder() \
                                        .with_task('') \
                                        .with_evaluators('') \
                                        .build()

    def test_missing_config_evaluators(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.EvalCommandConfig.builder() \
                                        .with_task('') \
                                        .with_scenes('') \
                                        .build()

    def test_no_config_error(self):
        try:
            rv.command.EvalCommandConfig.builder() \
                                        .with_task('') \
                                        .with_scenes('') \
                                        .with_evaluators('') \
                                        .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
