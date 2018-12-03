import unittest

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.core import Box

import tests.mock as mk


class PredictCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        task = rv.task.ChipClassificationConfig({})
        backend = rv.backend.KerasClassificationConfig('')
        with RVConfig.get_tmp_dir() as tmp_dir:
            cmd = rv.command.PredictCommandConfig.builder() \
                                                 .with_task(task) \
                                                 .with_root_uri(tmp_dir) \
                                                 .with_scenes('') \
                                                 .with_backend(backend) \
                                                 .build() \
                                                 .create_command()
            self.assertTrue(cmd, rv.command.PredictCommand)

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
        task = rv.task.ChipClassificationConfig({})
        backend = rv.backend.KerasClassificationConfig('')
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.PredictCommandConfig.builder() \
                                               .with_task(task) \
                                               .with_root_uri(tmp_dir) \
                                               .with_backend(backend) \
                                               .with_scenes(['']) \
                                               .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')

    def test_command_run_with_mocks(self):
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        backend_config = rv.BackendConfig.builder(mk.MOCK_BACKEND).build()
        backend = backend_config.create_backend(task_config)
        backend_config.mock.create_backend.return_value = backend
        task = task_config.create_task(backend)
        task_config.mock.create_task.return_value = task
        scene = mk.create_mock_scene()

        task.mock.get_predict_windows.return_value = [Box(0, 0, 1, 1)]

        cmd = rv.command.PredictCommandConfig.builder() \
                                             .with_task(task_config) \
                                             .with_backend(backend_config) \
                                             .with_scenes([scene]) \
                                             .with_root_uri('.') \
                                             .build() \
                                             .create_command()
        cmd.run()

        self.assertTrue(backend.mock.predict.called)


if __name__ == '__main__':
    unittest.main()
