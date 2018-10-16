import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig


class TestChipCommand(unittest.TestCase):
    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.ChipCommandConfig.builder() \
                                        .with_backend('') \
                                        .with_train_scenes('') \
                                        .with_val_scenes('') \
                                        .build()

    def test_missing_config_backend(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.ChipCommandConfig.builder() \
                                        .with_task('') \
                                        .with_train_scenes('') \
                                        .with_val_scenes('') \
                                        .build()

    def test_missing_config_train_scenes(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.ChipCommandConfig.builder() \
                                        .with_task('') \
                                        .with_backend('') \
                                        .with_val_scenes('') \
                                        .build()

    def test_missing_config_val_scenes(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.ChipCommandConfig.builder() \
                                        .with_task('') \
                                        .with_backend('') \
                                        .with_train_scenes('') \
                                        .build()

    def test_no_config_error(self):
        tmp = RVConfig.get_tmp_dir()
        try:
            rv.command.ChipCommandConfig.builder() \
                                        .with_task('') \
                                        .with_root_uri(tmp) \
                                        .with_backend('') \
                                        .with_train_scenes('') \
                                        .with_val_scenes('') \
                                        .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')


if __name__ == '__main__':
    unittest.main()
