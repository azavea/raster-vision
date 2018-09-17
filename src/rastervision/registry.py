import rastervision as rv
import rastervision.filesystem as rvfs

from rastervision.data.raster_source.default import (
    DefaultGeoTiffSourceProvider, DefaultImageSourceProvider)
from rastervision.data.label_source.default import (
    DefaultObjectDetectionGeoJSONSourceProvider,
    DefaultChipClassificationGeoJSONSourceProvider)
from rastervision.data.label_store.default import (
    DefaultObjectDetectionGeoJSONStoreProvider,
    DefaultChipClassificationGeoJSONStoreProvider)
from rastervision.evaluation.default import (
    DefaultObjectDetectioneEvaluatorProvider,
    DefaultChipClassificationEvaluatorProvider)
from typing import Union


class RegistryError(Exception):
    pass


class Registry:
    """Singleton that holds instances of Raster Vision types,
       to be referenced by configuration code by key.
    """

    def __init__(self):
        self._internal_config_builders = {
            # Tasks
            (rv.TASK, rv.OBJECT_DETECTION):
            rv.task.ObjectDetectionConfigBuilder,
            (rv.TASK, rv.CHIP_CLASSIFICATION):
            rv.task.ChipClassificationConfigBuilder,

            # Backends
            (rv.BACKEND, rv.TF_OBJECT_DETECTION):
            rv.backend.TFObjectDetectionConfigBuilder,
            (rv.BACKEND, rv.KERAS_CLASSIFICATION):
            rv.backend.KerasClassificationConfigBuilder,

            # Raster Transformers
            (rv.RASTER_TRANSFORMER, rv.STATS_TRANSFORMER):
            rv.data.StatsTransformerConfigBuilder,

            # Raster Sources
            (rv.RASTER_SOURCE, rv.GEOTIFF_SOURCE):
            rv.data.GeoTiffSourceConfigBuilder,
            (rv.RASTER_SOURCE, rv.IMAGE_SOURCE):
            rv.data.ImageSourceConfigBuilder,

            # Label Sources
            (rv.LABEL_SOURCE, rv.OBJECT_DETECTION_GEOJSON):
            rv.data.ObjectDetectionGeoJSONSourceConfigBuilder,
            (rv.LABEL_SOURCE, rv.CHIP_CLASSIFICATION_GEOJSON):
            rv.data.ChipClassificationGeoJSONSourceConfigBuilder,

            # Label Stores
            (rv.LABEL_STORE, rv.OBJECT_DETECTION_GEOJSON):
            rv.data.ObjectDetectionGeoJSONStoreConfigBuilder,
            (rv.LABEL_STORE, rv.CHIP_CLASSIFICATION_GEOJSON):
            rv.data.ChipClassificationGeoJSONStoreConfigBuilder,

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
        }

        self._internal_default_raster_sources = [
            DefaultGeoTiffSourceProvider,
            # This is the catch-all case, ensure it's on the bottom of the search stack.
            DefaultImageSourceProvider
        ]

        self._internal_default_label_sources = [
            DefaultObjectDetectionGeoJSONSourceProvider,
            DefaultChipClassificationGeoJSONSourceProvider
        ]

        self._internal_default_label_stores = [
            DefaultObjectDetectionGeoJSONStoreProvider,
            DefaultChipClassificationGeoJSONStoreProvider
        ]

        self._internal_default_evaluators = [
            DefaultObjectDetectioneEvaluatorProvider,
            DefaultChipClassificationEvaluatorProvider
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
            rv.AWS_BATCH: rv.runner.LocalExperimentRunner
        }

        self.filesystems = [
            rvfs.HttpFileSystem,
            rvfs.S3FileSystem,
            rvfs.LocalFileSystem
        ]

    def get_file_system(self, uri: str) -> rvfs.FileSystem:
        for fs in self.filesystems:
            if fs.matches_uri(uri):
                return fs
        return None

    def get_config_builder(self, group, key):
        internal_builder = self._internal_config_builders.get((group, key))
        if internal_builder:
            return internal_builder
        else:
            # TODO: Search plugins
            pass

        raise RegistryError('Unknown type {} for {} '.format(key, group))

    def get_default_raster_source_provider(self, s):
        """
        Gets the DefaultRasterSourceProvider for a given input string.
        """
        for provider in self._internal_default_raster_sources:
            if provider.handles(s):
                return provider

        # TODO: Search plugins

        raise RegistryError(
            'No DefaultRasterSourceProvider found for {}'.format(s))

    def get_default_label_source_provider(self, task_type, s):
        """
        Gets the DefaultRasterSourceProvider for a given input string.
        """
        for provider in self._internal_default_label_sources:
            if provider.handles(task_type, s):
                return provider

        # TODO: Search plugins

        raise RegistryError('No DefaultLabelSourceProvider '
                            'found for {} and task type {}'.format(
                                s, task_type))

    def get_default_label_store_provider(self, task_type, s=None):
        """
        Gets the DefaultRasterSourceProvider for a given input string.
        """

        # TODO: Search plugin before internal

        for provider in self._internal_default_label_stores:
            if s:
                if provider.handles(task_type, s):
                    return provider
            else:
                if provider.is_default_for(task_type):
                    return provider

        if s:
            raise RegistryError('No DefaultLabelStoreProvider '
                                'found for {} and task type {}'.format(
                                    s, task_type))
        else:
            raise RegistryError('No DefaultLabelStoreProvider '
                                'found for task type {}'.format(task_type))

    def get_default_evaluator_provider(self, task_type):
        """
        Gets the DefaultEvaluatorProvider for a given task
        """

        # TODO: Search plugin before internal

        for provider in self._internal_default_evaluators:
            if provider.is_default_for(task_type):
                return provider

        raise RegistryError('No DefaultEvaluatorProvider '
                            'found for task type {}'.format(task_type))

    def get_command_config_builder(self, command_type):
        builder = self.command_config_builders.get(command_type)
        if not builder:
            raise RegistryError(
                'No command found for type {}'.format(command_type))
        return builder

    def get_experiment_runner(self, runner_type):
        runner = self.experiment_runners.get(runner_type)
        if not runner:
            # TODO: Search plugins
            raise RegistryError(
                'No experiment runner for type {}'.format(runner_type))

        return runner()
