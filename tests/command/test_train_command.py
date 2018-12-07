import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig

import tests.mock as mk


class TrainCommand(mk.MockMixin, unittest.TestCase):
    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.TrainCommandConfig.builder() \
                                         .with_backend('') \
                                         .build()

    def test_missing_config_backend(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.TrainCommandConfig.builder() \
                                         .with_task('') \
                                         .build()

    def test_no_config_error(self):
        task = rv.task.ChipClassificationConfig({})
        backend = rv.backend.KerasClassificationConfig('')
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.TrainCommandConfig.builder() \
                                             .with_task(task) \
                                             .with_root_uri(tmp_dir) \
                                             .with_backend(backend) \
                                             .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')

    def test_command_run_with_mocks(self):
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        backend_config = rv.BackendConfig.builder(mk.MOCK_BACKEND).build()
        backend = backend_config.create_backend(task_config)
        backend_config.mock.create_backend.return_value = backend

        cmd = rv.command.TrainCommandConfig.builder() \
                                          .with_task(task_config) \
                                          .with_backend(backend_config) \
                                          .with_root_uri('.') \
                                          .build() \
                                          .create_command()
        cmd.run()

        self.assertTrue(backend.mock.train.called)


if __name__ == '__main__':
    unittest.main()
