# flake8: noqa


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


class MockMixin:
    def setUp(self):
        config = {'PLUGINS_modules': '["{}"]'.format(__name__)}
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
