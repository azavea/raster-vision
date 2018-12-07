import os
import unittest

import numpy as np

import rastervision as rv
from rastervision.rv_config import RVConfig
from rastervision.utils.misc import save_img

import tests.mock as mk


class TestAnalyzeCommand(mk.MockMixin, unittest.TestCase):
    def test_command_create(self):
        task = rv.task.ChipClassificationConfig({})
        with RVConfig.get_tmp_dir() as tmp_dir:
            img_path = os.path.join(tmp_dir, 'img.tif')
            chip = np.ones((2, 2, 4)).astype(np.uint8)
            chip[:, :, :] *= np.array([0, 1, 2, 3]).astype(np.uint8)
            save_img(chip, img_path)

            source = rv.data.ImageSourceConfig(img_path)

            scenes = [rv.data.SceneConfig('', source)]
            analyzers = [
                rv.analyzer.StatsAnalyzerConfig(stats_uri='dummy_path')
            ]

            cmd = rv.command.AnalyzeCommandConfig.builder() \
                                                 .with_task(task) \
                                                 .with_root_uri(tmp_dir) \
                                                 .with_scenes(scenes) \
                                                 .with_analyzers(analyzers) \
                                                 .build() \
                                                 .create_command()
            self.assertTrue(cmd, rv.command.AnalyzeCommand)

    def test_no_config_error(self):
        task = rv.task.ChipClassificationConfig({})
        try:
            with RVConfig.get_tmp_dir() as tmp_dir:
                rv.command.AnalyzeCommandConfig.builder() \
                                               .with_task(task) \
                                               .with_root_uri(tmp_dir) \
                                               .with_scenes(['']) \
                                               .with_analyzers(['']) \
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

    def test_command_run_with_mocks(self):
        task = rv.TaskConfig.builder(mk.MOCK_TASK).build()
        scene = mk.create_mock_scene()
        analyzer_config = rv.AnalyzerConfig.builder(mk.MOCK_ANALYZER).build()
        analyzer = analyzer_config.create_analyzer()
        analyzer_config.mock.create_analyzer.return_value = analyzer
        cmd = rv.command.AnalyzeCommandConfig.builder() \
                                             .with_task(task) \
                                             .with_scenes([scene]) \
                                             .with_root_uri('.') \
                                             .with_analyzers([analyzer_config]) \
                                             .build() \
                                             .create_command()
        cmd.run()

        self.assertTrue(analyzer.mock.process.called)


if __name__ == '__main__':
    unittest.main()
