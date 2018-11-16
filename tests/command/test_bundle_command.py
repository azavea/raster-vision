import os
import unittest
import zipfile

import rastervision as rv
from rastervision.command import BundleCommandConfig
from rastervision.protos.command_pb2 import CommandConfig as CommandConfigMsg
from rastervision.utils.files import (make_dir, load_json_config)
from rastervision.rv_config import RVConfig

import tests.mock as mk
from tests import data_file_path


class TestBundleCommand(mk.MockMixin, unittest.TestCase):
    def get_analyzer(self, tmp_dir):
        stats_uri = os.path.join(tmp_dir, 'stats.json')
        a = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                             .with_stats_uri(stats_uri) \
                             .build()
        return a

    def get_scene(self, tmp_dir):
        stats_uri = os.path.join(tmp_dir, 'stats.json')
        with open(stats_uri, 'w') as f:
            f.write('DUMMY')
        transformer = rv.RasterTransformerConfig \
                        .builder(rv.STATS_TRANSFORMER) \
                        .with_stats_uri(stats_uri) \
                        .build()

        raster_source = rv.RasterSourceConfig \
                          .builder(rv.IMAGE_SOURCE) \
                          .with_uri('TEST') \
                          .with_transformer(transformer) \
                          .build()

        scene = rv.SceneConfig.builder() \
                              .with_id('TEST') \
                              .with_raster_source(raster_source) \
                              .build()
        return scene

    def test_bundle_cc_command(self):
        def get_task(tmp_dir):
            predict_package_uri = os.path.join(tmp_dir, 'predict_package.zip')
            t = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                             .with_predict_package_uri(predict_package_uri) \
                             .with_classes(['class1']) \
                             .build()
            return t

        def get_backend(task, tmp_dir):
            model_uri = os.path.join(tmp_dir, 'model')
            with open(model_uri, 'w') as f:
                f.write('DUMMY')
            b = rv.BackendConfig.builder(rv.KERAS_CLASSIFICATION) \
                                .with_task(task) \
                                .with_model_defaults(rv.RESNET50_IMAGENET) \
                                .with_model_uri(model_uri) \
                                .build()
            return b

        with RVConfig.get_tmp_dir() as tmp_dir:
            task = get_task(tmp_dir)
            backend = get_backend(task, tmp_dir)
            analyzer = self.get_analyzer(tmp_dir)
            scene = self.get_scene(tmp_dir)
            cmd = BundleCommandConfig.builder() \
                                     .with_task(task) \
                                     .with_root_uri(tmp_dir) \
                                     .with_backend(backend) \
                                     .with_analyzers([analyzer]) \
                                     .with_scene(scene) \
                                     .build() \
                                     .create_command(tmp_dir)

            cmd.run(tmp_dir)

            package_dir = os.path.join(tmp_dir, 'package')
            make_dir(package_dir)
            with zipfile.ZipFile(task.predict_package_uri, 'r') as package_zip:
                package_zip.extractall(path=package_dir)

            bundle_config_path = os.path.join(package_dir,
                                              'bundle_config.json')
            bundle_config = load_json_config(bundle_config_path,
                                             CommandConfigMsg())

            self.assertEqual(bundle_config.command_type, rv.BUNDLE)

            actual = set(os.listdir(package_dir))
            expected = set(['stats.json', 'model', 'bundle_config.json'])

            self.assertEqual(actual, expected)

    def test_bundle_od_command(self):
        def get_task(tmp_dir):
            predict_package_uri = os.path.join(tmp_dir, 'predict_package.zip')
            t = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                             .with_predict_package_uri(predict_package_uri) \
                             .with_classes(['class1']) \
                             .build()
            return t

        def get_backend(task, tmp_dir):
            model_uri = os.path.join(tmp_dir, 'model')
            template_uri = data_file_path(
                'tf_object_detection/embedded_ssd_mobilenet_v1_coco.config')
            with open(model_uri, 'w') as f:
                f.write('DUMMY')
            b = rv.BackendConfig.builder(rv.TF_OBJECT_DETECTION) \
                                .with_task(task) \
                                .with_template(template_uri) \
                                .with_model_uri(model_uri) \
                                .build()
            return b

        with RVConfig.get_tmp_dir() as tmp_dir:
            task = get_task(tmp_dir)
            backend = get_backend(task, tmp_dir)
            analyzer = self.get_analyzer(tmp_dir)
            scene = self.get_scene(tmp_dir)
            cmd = BundleCommandConfig.builder() \
                                     .with_task(task) \
                                     .with_root_uri(tmp_dir) \
                                     .with_backend(backend) \
                                     .with_analyzers([analyzer]) \
                                     .with_scene(scene) \
                                     .build() \
                                     .create_command()

            cmd.run(tmp_dir)

            package_dir = os.path.join(tmp_dir, 'package')
            make_dir(package_dir)
            with zipfile.ZipFile(task.predict_package_uri, 'r') as package_zip:
                package_zip.extractall(path=package_dir)

            bundle_config_path = os.path.join(package_dir,
                                              'bundle_config.json')
            bundle_config = load_json_config(bundle_config_path,
                                             CommandConfigMsg())

            self.assertEqual(bundle_config.command_type, rv.BUNDLE)

            actual = set(os.listdir(package_dir))
            expected = set(['stats.json', 'model', 'bundle_config.json'])

            self.assertEqual(actual, expected)

    def test_missing_config_task(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.BundleCommandConfig.builder() \
                                          .with_scene('') \
                                          .with_backend('') \
                                          .with_analyzers([]) \
                                          .build()

    def test_missing_config_backendf(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.BundleCommandConfig.builder() \
                                          .with_task('') \
                                          .with_scene('') \
                                          .with_analyzers([]) \
                                          .build()

    def test_missing_config_scene(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.BundleCommandConfig.builder() \
                                          .with_task('') \
                                          .with_backend('') \
                                          .with_analyzers([]) \
                                          .build()

    def test_missing_config_analyzers(self):
        with self.assertRaises(rv.ConfigError):
            rv.command.BundleCommandConfig.builder() \
                                          .with_task('') \
                                          .with_scene('') \
                                          .with_backend('') \
                                          .build()

    def test_command_run_with_mocks(self):
        with RVConfig.get_tmp_dir() as tmp_dir:
            task_config = rv.TaskConfig.builder(mk.MOCK_TASK).build()
            task_config.predict_package_uri = os.path.join(
                tmp_dir, 'predict_package.zip')
            backend_config = rv.BackendConfig.builder(mk.MOCK_BACKEND).build()
            scene = mk.create_mock_scene()
            analyzer_config = rv.AnalyzerConfig.builder(
                mk.MOCK_ANALYZER).build()

            cmd = rv.command.BundleCommandConfig.builder() \
                                              .with_task(task_config) \
                                              .with_backend(backend_config) \
                                              .with_scene(scene) \
                                              .with_analyzers([analyzer_config]) \
                                              .with_root_uri('.') \
                                              .build() \
                                              .create_command()
            cmd.run()

            self.assertTrue(os.path.exists(task_config.predict_package_uri))
            self.assertTrue(analyzer_config.mock.save_bundle_files.called)


if __name__ == '__main__':
    unittest.main()
