import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig


class TestEvalCommand(unittest.TestCase):
    def test_command_create(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            cmd = rv.command.EvalCommandConfig.builder() \
                                              .with_task('') \
                                              .with_root_uri(tmp_dir) \
                                              .with_scenes('') \
                                              .with_evaluators('') \
                                              .build() \
                                              .create_command()
            self.assertTrue(cmd, rv.command.EvalCommand)

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
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.EvalCommandConfig.builder() \
                                            .with_task('') \
                                            .with_root_uri(tmp_dir) \
                                            .with_scenes('') \
                                            .with_evaluators('') \
                                            .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
