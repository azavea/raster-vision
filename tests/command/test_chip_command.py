import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig

import tests.mock as mk


class TestChipCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        task = rv.task.ChipClassificationConfig({})
        backend = rv.backend.KerasClassificationConfig('')
        with RVConfig.get_tmp_dir() as tmp_dir:
            cmd = rv.command.ChipCommandConfig.builder() \
                                              .with_task(task) \
                                              .with_backend(backend) \
                                              .with_train_scenes('') \
                                              .with_val_scenes('') \
                                              .with_root_uri(tmp_dir) \
                                              .build() \
                                              .create_command()
            self.assertTrue(cmd, rv.command.ChipCommand)

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
        task = rv.task.ChipClassificationConfig({})
        backend = rv.backend.KerasClassificationConfig('')
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.ChipCommandConfig.builder() \
                                            .with_task(task) \
                                            .with_root_uri(tmp_dir) \
                                            .with_backend(backend) \
                                            .with_train_scenes('') \
                                            .with_val_scenes('') \
                                            .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')

    def test_command_run_with_mocks(self):
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        backend_config = rv.BackendConfig.builder(mk.MOCK_BACKEND).build()
        backend = backend_config.create_backend(task_config)
        task = task_config.create_task(backend)
        task_config.mock.create_task.return_value = task
        backend_config.mock.create_backend.return_value = backend
        scene = mk.create_mock_scene()
        cmd = rv.command.ChipCommandConfig.builder() \
                                          .with_task(task_config) \
                                          .with_backend(backend_config) \
                                          .with_train_scenes([scene]) \
                                          .with_val_scenes([scene]) \
                                          .with_root_uri('.') \
                                          .build() \
                                          .create_command()
        cmd.run()

        self.assertTrue(task.mock.get_train_windows.called)
        self.assertTrue(backend.mock.process_sceneset_results.called)


if __name__ == '__main__':
    unittest.main()
