import os
import unittest

import numpy as np

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.utils.misc import save_img

import tests.mock as mk


class TestEvalCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        task = rv.task.ChipClassificationConfig({})
        with RVConfig.get_tmp_dir() as tmp_dir:
            img_path = os.path.join(tmp_dir, 'img.tif')
            chip = np.ones((2, 2, 4)).astype(np.uint8)
            chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
            save_img(chip, img_path)

            source = rv.data.ImageSourceConfig(img_path)

            scenes = [rv.data.SceneConfig('', source)]
            analyzers = [rv.evaluation.ObjectDetectionEvaluatorConfig({})]

            cmd = rv.command.EvalCommandConfig.builder() \
                                              .with_task(task) \
                                              .with_root_uri(tmp_dir) \
                                              .with_scenes(scenes) \
                                              .with_evaluators(analyzers) \
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
        task = rv.task.ChipClassificationConfig({})
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.EvalCommandConfig.builder() \
                                            .with_task(task) \
                                            .with_root_uri(tmp_dir) \
                                            .with_scenes(['']) \
                                            .with_evaluators(['']) \
                                            .build()
        except rv.ConfigError:
            self.fail('rv.ConfigError raised unexpectedly')

    def test_command_run_with_mocks(self):
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        scene = mk.create_mock_scene()
        evaluator_config = rv.EvaluatorConfig.builder(
            mk.MOCK_EVALUATOR).build()
        evaluator = evaluator_config.create_evaluator()
        evaluator_config.mock.create_evaluator.return_value = evaluator

        cmd = rv.command.EvalCommandConfig.builder() \
                                          .with_task(task_config) \
                                          .with_scenes([scene]) \
                                          .with_evaluators([evaluator_config]) \
                                          .with_root_uri('.') \
                                          .build() \
                                          .create_command()
        cmd.run()

        self.assertTrue(evaluator.mock.process.called)


if __name__ == '__main__':
    unittest.main()
