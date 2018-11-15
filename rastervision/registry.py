import rastervision as rv
import rastervision.filesystem as rvfs
from rastervision.cli import Verbosity
from rastervision.rv_config import RVConfig
from rastervision.plugin import PluginRegistry
from rastervision.data.raster_source.default import (
    GeoTiffSourceDefaultProvider, ImageSourceDefaultProvider)
from rastervision.data.vector_source.default import (
    GeoJSONVectorSourceDefaultProvider, MBTilesVectorSourceDefaultProvider)
from rastervision.data.label_source.default import (
    ObjectDetectionLabelSourceDefaultProvider,
    ChipClassificationLabelSourceDefaultProvider,
    SemanticSegmentationLabelSourceDefaultProvider)
from rastervision.data.label_store.default import (
    ObjectDetectionGeoJSONStoreDefaultProvider,
    ChipClassificationGeoJSONStoreDefaultProvider,
    SemanticSegmentationRasterStoreDefaultProvider)
from rastervision.evaluation.default import (
    ObjectDetectionEvaluatorDefaultProvider,
    ChipClassificationEvaluatorDefaultProvider,
    SemanticSegmentationEvaluatorDefaultProvider)


class RegistryError(Exception):
    pass


class Registry:
    """Singleton that holds instances of Raster Vision types,
       to be referenced by configuration code by key.
    """

    def __init__(self):
        self._rv_config = None
        self._plugin_registry = None

        self._internal_config_builders = {
            # Tasks
            (rv.TASK, rv.OBJECT_DETECTION):
            rv.task.ObjectDetectionConfigBuilder,
            (rv.TASK, rv.CHIP_CLASSIFICATION):
            rv.task.ChipClassificationConfigBuilder,
            (rv.TASK, rv.SEMANTIC_SEGMENTATION):
            rv.task.SemanticSegmentationConfigBuilder,

            # Backends
            (rv.BACKEND, rv.TF_OBJECT_DETECTION):
            rv.backend.TFObjectDetectionConfigBuilder,
            (rv.BACKEND, rv.KERAS_CLASSIFICATION):
            rv.backend.KerasClassificationConfigBuilder,
            (rv.BACKEND, rv.TF_DEEPLAB):
            rv.backend.TFDeeplabConfigBuilder,

            # Raster Transformers
            (rv.RASTER_TRANSFORMER, rv.STATS_TRANSFORMER):
            rv.data.StatsTransformerConfigBuilder,

            # Raster Sources
            (rv.RASTER_SOURCE, rv.GEOTIFF_SOURCE):
            rv.data.GeoTiffSourceConfigBuilder,
            (rv.RASTER_SOURCE, rv.RASTERIZED_SOURCE):
            rv.data.RasterizedSourceConfigBuilder,
            (rv.RASTER_SOURCE, rv.IMAGE_SOURCE):
            rv.data.ImageSourceConfigBuilder,

            # Alias provided for backwards compatibility.
            (rv.RASTER_SOURCE, rv.GEOJSON_SOURCE):
            rv.data.RasterizedSourceConfigBuilder,

            # Vector Sources
            (rv.VECTOR_SOURCE, rv.MBTILES_SOURCE):
            rv.data.MBTilesVectorSourceConfigBuilder,
            (rv.VECTOR_SOURCE, rv.GEOJSON_SOURCE):
            rv.data.GeoJSONVectorSourceConfigBuilder,

            # Label Sources
            (rv.LABEL_SOURCE, rv.CHIP_CLASSIFICATION):
            rv.data.ChipClassificationLabelSourceConfigBuilder,
            (rv.LABEL_SOURCE, rv.OBJECT_DETECTION):
            rv.data.ObjectDetectionLabelSourceConfigBuilder,
            (rv.LABEL_SOURCE, rv.SEMANTIC_SEGMENTATION):
            rv.data.SemanticSegmentationLabelSourceConfigBuilder,

            # Label Source aliases provided for backward-compatibility
            (rv.LABEL_SOURCE, rv.OBJECT_DETECTION_GEOJSON):
            rv.data.ObjectDetectionLabelSourceConfigBuilder,
            (rv.LABEL_SOURCE, rv.CHIP_CLASSIFICATION_GEOJSON):
            rv.data.ChipClassificationLabelSourceConfigBuilder,
            (rv.LABEL_SOURCE, rv.SEMANTIC_SEGMENTATION_RASTER):
            rv.data.SemanticSegmentationLabelSourceConfigBuilder,

            # Label Stores
            (rv.LABEL_STORE, rv.OBJECT_DETECTION_GEOJSON):
            rv.data.ObjectDetectionGeoJSONStoreConfigBuilder,
            (rv.LABEL_STORE, rv.CHIP_CLASSIFICATION_GEOJSON):
            rv.data.ChipClassificationGeoJSONStoreConfigBuilder,
            (rv.LABEL_STORE, rv.SEMANTIC_SEGMENTATION_RASTER):
            rv.data.SemanticSegmentationRasterStoreConfigBuilder,

            # Analyzers
            (rv.ANALYZER, rv.STATS_ANALYZER):
            rv.analyzer.StatsAnalyzerConfigBuilder,

            # Augmentors
            (rv.AUGMENTOR, rv.NODATA_AUGMENTOR):
            rv.augmentor.NodataAugmentorConfigBuilder,

            # Evaluators
            (rv.EVALUATOR, rv.CHIP_CLASSIFICATION_EVALUATOR):
            rv.evaluation.ChipClassificationEvaluatorConfigBuilder,
            (rv.EVALUATOR, rv.OBJECT_DETECTION_EVALUATOR):
            rv.evaluation.ObjectDetectionEvaluatorConfigBuilder,
            (rv.EVALUATOR, rv.SEMANTIC_SEGMENTATION_EVALUATOR):
            rv.evaluation.SemanticSegmentationEvaluatorConfigBuilder,
        }

        self._internal_default_raster_sources = [
            GeoTiffSourceDefaultProvider,
            # This is the catch-all case, ensure it's on the bottom of the search stack.
            ImageSourceDefaultProvider
        ]

        self._internal_default_vector_sources = [
            GeoJSONVectorSourceDefaultProvider,
            MBTilesVectorSourceDefaultProvider
        ]

        self._internal_default_label_sources = [
            ObjectDetectionLabelSourceDefaultProvider,
            ChipClassificationLabelSourceDefaultProvider,
            SemanticSegmentationLabelSourceDefaultProvider
        ]

        self._internal_default_label_stores = [
            ObjectDetectionGeoJSONStoreDefaultProvider,
            ChipClassificationGeoJSONStoreDefaultProvider,
            SemanticSegmentationRasterStoreDefaultProvider
        ]

        self._internal_default_evaluators = [
            ObjectDetectionEvaluatorDefaultProvider,
            ChipClassificationEvaluatorDefaultProvider,
            SemanticSegmentationEvaluatorDefaultProvider
        ]

        self.command_config_builders = {
            rv.ANALYZE: rv.command.AnalyzeCommandConfigBuilder,
            rv.CHIP: rv.command.ChipCommandConfigBuilder,
            rv.TRAIN: rv.command.TrainCommandConfigBuilder,
            rv.PREDICT: rv.command.PredictCommandConfigBuilder,
            rv.EVAL: rv.command.EvalCommandConfigBuilder,
            rv.BUNDLE: rv.command.BundleCommandConfigBuilder
        }

        self.experiment_runners = {
            rv.LOCAL: rv.runner.LocalExperimentRunner,
            rv.AWS_BATCH: rv.runner.AwsBatchExperimentRunner
        }

        self.filesystems = [
            rvfs.HttpFileSystem,
            rvfs.S3FileSystem,
            # This is the catch-all case, ensure it's on the bottom of the search stack.
            rvfs.LocalFileSystem
        ]

    def initialize_config(self,
                          profile=None,
                          rv_home=None,
                          config_overrides=None,
                          verbosity=Verbosity.NORMAL):
        self._rv_config = RVConfig(
            profile=profile,
            rv_home=rv_home,
            config_overrides=config_overrides,
            verbosity=verbosity)
        # Reset the plugins in case this is a re-initialization,
        self._plugin_registry = None

    def _get_rv_config(self):
        """Returns the application configuration"""
        if self._rv_config is None:
            self.initialize_config()
        return self._rv_config

    def _ensure_plugins_loaded(self):
        if not self._plugin_registry:
            self._load_plugins()

    def _get_plugin_registry(self):
        self._ensure_plugins_loaded()
        return self._plugin_registry

    def _load_plugins(self):
        rv_config = self._get_rv_config()
        plugin_config = rv_config.get_subconfig('PLUGINS')
        self._plugin_registry = PluginRegistry(plugin_config,
                                               rv_config.rv_home)

    def get_config_builder(self, group, key):
        internal_builder = self._internal_config_builders.get((group, key))
        if internal_builder:
            return internal_builder
        else:
            self._ensure_plugins_loaded()
            plugin_builder = self._plugin_registry.config_builders.get((group,
                                                                        key))
            if plugin_builder:
                return plugin_builder

        raise RegistryError('Unknown type {} for {} '.format(key, group))

    def get_file_system(self, uri: str, mode: str = 'r',
                        search_plugins=True) -> rvfs.FileSystem:
        # If we are currently loading plugins, don't search for
        # plugin filesystems.
        if search_plugins:
            self._ensure_plugins_loaded()
            filesystems = (
                self._plugin_registry.filesystems + self.filesystems)
        else:
            filesystems = self.filesystems

        for fs in filesystems:
            if fs.matches_uri(uri, mode):
                return fs
        if mode == 'w':
            raise RegistryError('No matching filesystem to handle '
                                'writing to uri {}'.format(uri))
        else:
            raise RegistryError('No matching filesystem to handle '
                                'reading from uri {}'.format(uri))

    def get_raster_source_default_provider(self, s):
        """
        Gets the RasterSourceDefaultProvider for a given input string.
        """
        self._ensure_plugins_loaded()
        providers = (self._plugin_registry.default_raster_sources +
                     self._internal_default_raster_sources)

        for provider in providers:
            if provider.handles(s):
                return provider

        raise RegistryError(
            'No RasterSourceDefaultProvider found for {}'.format(s))

    def get_vector_source_default_provider(self, s):
        """
        Gets the VectorSourceDefaultProvider for a given input string.
        """
        self._ensure_plugins_loaded()
        providers = (self._plugin_registry.default_vector_sources +
                     self._internal_default_vector_sources)

        for provider in providers:
            if provider.handles(s):
                return provider

        raise RegistryError(
            'No VectorSourceDefaultProvider found for {}'.format(s))

    def get_label_source_default_provider(self, task_type, s):
        """
        Gets the RasterSourceDefaultProvider for a given input string.
        """
        self._ensure_plugins_loaded()
        providers = (self._plugin_registry.default_label_sources +
                     self._internal_default_label_sources)

        for provider in providers:
            if provider.handles(task_type, s):
                return provider

        raise RegistryError('No LabelSourceDefaultProvider '
                            'found for {} and task type {}'.format(
                                s, task_type))

    def get_label_store_default_provider(self, task_type, s=None):
        """
        Gets the RasterSourceDefaultProvider for a given input string.
        """

        self._ensure_plugins_loaded()
        providers = (self._plugin_registry.default_label_stores +
                     self._internal_default_label_stores)

        for provider in providers:
            if s:
                if provider.handles(task_type, s):
                    return provider
            else:
                if provider.is_default_for(task_type):
                    return provider

        if s:
            raise RegistryError('No LabelStoreDefaultProvider '
                                'found for {} and task type {}'.format(
                                    s, task_type))
        else:
            raise RegistryError('No LabelStoreDefaultProvider '
                                'found for task type {}'.format(task_type))

    def get_evaluator_default_provider(self, task_type):
        """
        Gets the EvaluatorDefaultProvider for a given task
        """

        self._ensure_plugins_loaded()
        providers = (self._plugin_registry.default_evaluators +
                     self._internal_default_evaluators)

        for provider in providers:
            if provider.is_default_for(task_type):
                return provider

        raise RegistryError('No EvaluatorDefaultProvider '
                            'found for task type {}'.format(task_type))

    def get_command_config_builder(self, command_type):
        builder = self.command_config_builders.get(command_type)
        if not builder:
            raise RegistryError(
                'No command found for type {}'.format(command_type))
        return builder

    def get_experiment_runner(self, runner_type):
        internal_runner = self.experiment_runners.get(runner_type)
        if internal_runner:
            return internal_runner()
        else:
            self._ensure_plugins_loaded()
            plugin_runner = self._plugin_registry.experiment_runners.get(
                runner_type)
            if plugin_runner:
                return plugin_runner()
        raise RegistryError(
            'No experiment runner for type {}'.format(runner_type))

    def get_experiment_runner_keys(self):
        self._ensure_plugins_loaded()
        return (list(self.experiment_runners.keys()) + list(
            self._plugin_registry.experiment_runners.keys()))
