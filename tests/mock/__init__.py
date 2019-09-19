# flake8: noqa

import rastervision as rv


class SupressDeepCopyMixin:
    """Supress deep copy in mock objects, since we want to check mocks after processing."""
    def __deepcopy__(self, memodict={}):
        return self


from tests.mock.task import *
from tests.mock.backend import *
from tests.mock.raster_source import *
from tests.mock.label_source import *
from tests.mock.label_store import *
from tests.mock.raster_transformer import *
from tests.mock.augmentor import *
from tests.mock.analyzer import *
from tests.mock.evaluator import *
from tests.mock.command import *
from tests.mock.aux_command import *


class MockMixin:
    def mock_config(self):
        return {'PLUGINS_modules': '["{}"]'.format(__name__)}

    def setUp(self):
        config = self.mock_config()
        rv._registry.initialize_config(config_overrides=config)
        super().setUp()

    def tearDown(self):
        rv._registry.initialize_config()
        super().tearDown()


def create_mock_scene():
    raster_transformer_config = rv.RasterTransformerConfig.builder(MOCK_TRANSFORMER) \
                                                          .build()

    raster_source_config = rv.RasterSourceConfig.builder(MOCK_SOURCE) \
                                                .with_transformer(raster_transformer_config) \
                                                .build()

    label_source_config = rv.LabelSourceConfig.builder(MOCK_SOURCE) \
                                                .build()

    label_store_config = rv.LabelStoreConfig.builder(MOCK_STORE) \
                                                .build()

    return rv.SceneConfig.builder() \
                         .with_id('test') \
                         .with_raster_source(raster_source_config) \
                         .with_label_source(label_source_config) \
                         .with_label_store(label_store_config) \
                         .build()


def create_mock_experiment():
    b = rv.BackendConfig.builder(MOCK_BACKEND).build()
    t = rv.TaskConfig.builder(MOCK_TASK).build()
    ds = rv.DatasetConfig.builder() \
                         .with_test_scenes([create_mock_scene()]) \
                         .with_validation_scenes([create_mock_scene()]) \
                         .build()
    e = rv.EvaluatorConfig.builder(MOCK_EVALUATOR).build()

    return rv.ExperimentConfig.builder() \
                              .with_backend(b) \
                              .with_task(t) \
                              .with_dataset(ds) \
                              .with_evaluator(e) \
                              .with_root_uri('/dev/null') \
                              .with_id('mock_experiment') \
                              .build()


def register_plugin(plugin_registry):
    plugin_registry.register_config_builder(rv.TASK, MOCK_TASK,
                                            MockTaskConfigBuilder)
    plugin_registry.register_config_builder(rv.BACKEND, MOCK_BACKEND,
                                            MockBackendConfigBuilder)
    plugin_registry.register_config_builder(rv.RASTER_SOURCE, MOCK_SOURCE,
                                            MockRasterSourceConfigBuilder)
    plugin_registry.register_config_builder(rv.LABEL_SOURCE, MOCK_SOURCE,
                                            MockLabelSourceConfigBuilder)
    plugin_registry.register_config_builder(rv.LABEL_STORE, MOCK_STORE,
                                            MockLabelStoreConfigBuilder)
    plugin_registry.register_config_builder(
        rv.RASTER_TRANSFORMER, MOCK_TRANSFORMER,
        MockRasterTransformerConfigBuilder)
    plugin_registry.register_config_builder(rv.AUGMENTOR, MOCK_AUGMENTOR,
                                            MockAugmentorConfigBuilder)
    plugin_registry.register_config_builder(rv.ANALYZER, MOCK_ANALYZER,
                                            MockAnalyzerConfigBuilder)
    plugin_registry.register_config_builder(rv.EVALUATOR, MOCK_EVALUATOR,
                                            MockEvaluatorConfigBuilder)

    plugin_registry.register_command_config_builder(MOCK_COMMAND,
                                                    MockCommandConfigBuilder)
    plugin_registry.register_aux_command(MOCK_AUX_COMMAND, MockAuxCommand)
