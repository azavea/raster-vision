import os
import unittest

import rastervision as rv

from tests import data_file_path


class TestPlugin(unittest.TestCase):
    def test_single_plugin_from_path(self):
        config = {
            'PLUGINS_files':
            '["{}"]'.format(data_file_path('plugins/noop_augmentor.py'))
        }
        rv._registry.initialize_config(config_overrides=config)

        try:
            augmentor = rv.AugmentorConfig.builder('NOOP_AUGMENTOR') \
                                          .build() \
                                          .create_augmentor()

            self.assertIsInstance(augmentor, rv.augmentor.Augmentor)
        finally:
            # Reset config
            rv._registry.initialize_config()

    def test_plugin_from_module(self):
        config = {'PLUGINS_modules': '["{}"]'.format(__name__)}
        rv._registry.initialize_config(config_overrides=config)

        try:
            augmentor = rv.AnalyzerConfig.builder('NOOP_ANALYZER') \
                                         .build() \
                                         .create_analyzer()

            self.assertIsInstance(augmentor, rv.analyzer.Analyzer)
        finally:
            # Reset config
            rv._registry.initialize_config()

    def test_runs_noop_experiment_from_plugins(self):
        # set the env var to have rv pick up this configuration
        # which adds the tests.test_plugin module as a plugin.
        old_rv_config = os.environ.get('RV_CONFIG')
        os.environ['RV_CONFIG'] = data_file_path('plugins/default')

        try:
            plugin_files = [
                data_file_path('plugins/noop_augmentor.py'),
                data_file_path('plugins/noop_task.py'),
                data_file_path('plugins/noop_backend.py'),
                data_file_path('plugins/noop_raster_transformer.py'),
                data_file_path('plugins/noop_raster_source.py'),
                data_file_path('plugins/noop_label_source.py'),
                data_file_path('plugins/noop_label_store.py'),
                data_file_path('plugins/noop_evaluator.py'),
                data_file_path('plugins/noop_runner.py')
            ]

            config = {
                'PLUGINS_files': '["{}"]'.format('","'.join(plugin_files))
            }
            rv._registry.initialize_config(config_overrides=config)

            # Check proto serialization

            msg = rv.AnalyzerConfig.builder('NOOP_ANALYZER') \
                                   .build() \
                                   .to_proto()
            analyzer = rv.AnalyzerConfig.from_proto(msg)

            msg = rv.AugmentorConfig.builder('NOOP_AUGMENTOR') \
                                   .build() \
                                   .to_proto()
            augmentor = rv.AugmentorConfig.from_proto(msg)

            msg = rv.TaskConfig.builder('NOOP_TASK') \
                               .build() \
                               .to_proto()
            task = rv.TaskConfig.from_proto(msg)

            msg = rv.BackendConfig.builder('NOOP_BACKEND') \
                                  .build() \
                                  .to_proto()
            backend = rv.BackendConfig.from_proto(msg)

            msg = rv.RasterTransformerConfig.builder('NOOP_TRANSFORMER') \
                                            .build() \
                                            .to_proto()
            raster_transformer = rv.RasterTransformerConfig.from_proto(msg)

            msg = rv.RasterSourceConfig.builder('NOOP_SOURCE') \
                                       .with_transformer(raster_transformer) \
                                       .build() \
                                       .to_proto()
            raster_source = rv.RasterSourceConfig.from_proto(msg)

            msg = rv.LabelSourceConfig.builder('NOOP_SOURCE') \
                                       .build() \
                                       .to_proto()
            label_source = rv.LabelSourceConfig.from_proto(msg)

            msg = rv.LabelStoreConfig.builder('NOOP_STORE') \
                                     .build() \
                                     .to_proto()
            label_store = rv.LabelStoreConfig.from_proto(msg)

            msg = rv.EvaluatorConfig.builder('NOOP_EVALUATOR') \
                                   .build() \
                                   .to_proto()
            evaluator = rv.EvaluatorConfig.from_proto(msg)

            train_scene = rv.SceneConfig.builder() \
                                        .with_id('train') \
                                        .with_raster_source(raster_source) \
                                        .with_label_source(label_source) \
                                        .build()

            val_scene = rv.SceneConfig.builder() \
                                      .with_id('val') \
                                      .with_raster_source(raster_source) \
                                      .with_label_source(label_source) \
                                      .with_label_store(label_store) \
                                      .build()

            dataset = rv.DatasetConfig.builder() \
                                      .with_train_scene(train_scene) \
                                      .with_validation_scene(val_scene) \
                                      .with_augmentor(augmentor) \
                                      .build()

            e = rv.ExperimentConfig.builder() \
                                   .with_id('plugin_test') \
                                   .with_task(task) \
                                   .with_backend(backend) \
                                   .with_dataset(dataset) \
                                   .with_analyzer(analyzer) \
                                   .with_evaluator(evaluator) \
                                   .with_root_uri('/no/matter') \
                                   .build()

            rv.ExperimentRunner.get_runner('NOOP_RUNNER').run(
                e, rerun_commands=True)
        finally:
            # Reset environment var
            if old_rv_config:
                os.environ['RV_CONFIG'] = old_rv_config
            else:
                del os.environ['RV_CONFIG']


if __name__ == '__main__':
    unittest.main()

from rastervision.analyzer import (Analyzer, AnalyzerConfig,
                                   AnalyzerConfigBuilder)  # noqa
from rastervision.protos.analyzer_pb2 \
    import AnalyzerConfig as AnalyzerConfigMsg # noqa

NOOP_ANALYZER = 'NOOP_ANALYZER'


class NoopAnalyzer(Analyzer):
    def process(self, training_data, tmp_dir):
        pass


class NoopAnalyzerConfig(AnalyzerConfig):
    def __init__(self):
        super().__init__(NOOP_ANALYZER)

    def to_proto(self):
        return AnalyzerConfigMsg(analyzer_type=self.analyzer_type)

    def create_analyzer(self):
        return NoopAnalyzer()

    def update_for_command(self,
                           command_type,
                           experiment_config,
                           context=None,
                           io_def=None):
        return io_def

    def save_bundle_files(self, bundle_dir):
        return (self, [])

    def load_bundle_files(self, bundle_dir):
        return self


class NoopAnalyzerConfigBuilder(AnalyzerConfigBuilder):
    def __init__(self, prev=None):
        super().__init__(NoopAnalyzerConfig, {})

    def from_proto(self, msg):
        return NoopAnalyzerConfigBuilder()


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.ANALYZER, NOOP_ANALYZER,
                                            NoopAnalyzerConfigBuilder)
