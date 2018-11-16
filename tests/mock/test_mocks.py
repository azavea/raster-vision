import unittest
import tempfile

import rastervision as rv
from rastervision.core import Box

import tests.mock as mk


class TestPlugin(mk.MockMixin, unittest.TestCase):
    def test_mocks(self):
        """Test to ensure all mocks are working as expected."""
        task_config = rv.TaskConfig.builder(mk.MOCK_TASK) \
                                   .build()

        backend_config = rv.BackendConfig.builder(mk.MOCK_BACKEND) \
                                         .build()

        raster_transformer_config = rv.RasterTransformerConfig.builder(
            mk.MOCK_TRANSFORMER).build()

        raster_source_config = rv.RasterSourceConfig.builder(mk.MOCK_SOURCE) \
                                                    .with_transformer(
                                                        raster_transformer_config) \
                                                    .build()

        label_source_config = rv.LabelSourceConfig.builder(mk.MOCK_SOURCE) \
                                                    .build()

        label_store_config = rv.LabelStoreConfig.builder(mk.MOCK_STORE) \
                                                    .build()

        scene_config = rv.SceneConfig.builder() \
                                     .with_id('test') \
                                     .with_raster_source(raster_source_config) \
                                     .with_label_source(label_source_config) \
                                     .with_label_store(label_store_config) \
                                     .build()

        augmentor_config = rv.AugmentorConfig.builder(mk.MOCK_AUGMENTOR) \
                                             .build()

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scene(scene_config) \
                                  .with_validation_scene(scene_config) \
                                  .with_augmentor(augmentor_config)  \
                                  .build()

        analyzer_config = rv.AnalyzerConfig.builder(mk.MOCK_ANALYZER).build()
        evaluator_config = rv.EvaluatorConfig.builder(
            mk.MOCK_EVALUATOR).build()

        # Create entities from configuration

        backend = backend_config.create_backend(task_config)
        task = task_config.create_task(backend)
        scene = scene_config.create_scene(task_config, '.')
        _ = augmentor_config.create_augmentor()  # noqa
        _ = analyzer_config.create_analyzer()  # noqa
        _ = evaluator_config.create_evaluator()  # noqa

        # Assert some things

        task_config.mock.create_task.assert_called_once_with(backend)
        self.assertEqual(task.get_predict_windows(Box(0, 0, 1, 1)), [])

        _ = scene.raster_source.get_chip(Box(0, 0, 1, 1))  # noqa
        self.assertTrue(
            scene.raster_source.raster_transformers[0].mock.transform.called)

        # Create and run experiment

        with tempfile.TemporaryDirectory() as tmp_dir:
            e = rv.ExperimentConfig.builder() \
                                   .with_task(task_config) \
                                   .with_backend(backend_config) \
                                   .with_dataset(dataset) \
                                   .with_analyzer(analyzer_config) \
                                   .with_evaluator(evaluator_config) \
                                   .with_root_uri(tmp_dir) \
                                   .with_id('test') \
                                   .build()

            rv.ExperimentRunner.get_runner(rv.LOCAL).run(e)
