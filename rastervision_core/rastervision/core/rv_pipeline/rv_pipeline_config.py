from os.path import join
from typing import List, TYPE_CHECKING, Optional

from rastervision.pipeline.pipeline_config import PipelineConfig
from rastervision.core.data import (DatasetConfig, StatsTransformerConfig,
                                    LabelStoreConfig, SceneConfig)
from rastervision.core.utils.misc import Proportion
from rastervision.core.analyzer import StatsAnalyzerConfig
from rastervision.core.backend import BackendConfig
from rastervision.core.evaluation import EvaluatorConfig
from rastervision.core.analyzer import AnalyzerConfig
from rastervision.pipeline.config import (register_config, Field, Config)

if TYPE_CHECKING:
    from rastervision.core.backend.backend import Backend  # noqa


@register_config('predict_options')
class PredictOptions(Config):
    # TODO: predict_chip_sz and predict_batch_sz should probably be moved here
    pass


@register_config('rv_pipeline')
class RVPipelineConfig(PipelineConfig):
    """Configure an :class:`.RVPipeline`."""

    dataset: DatasetConfig = Field(
        ...,
        description=
        'Dataset containing train, validation, and optional test scenes.')
    backend: BackendConfig = Field(
        ..., description='Backend to use for interfacing with ML library.')
    evaluators: List[EvaluatorConfig] = Field(
        [],
        description=(
            'Evaluators to run during analyzer command. If list is empty '
            'the default evaluator is added.'))
    analyzers: List[AnalyzerConfig] = Field(
        [],
        description=
        ('Analyzers to run during analyzer command. A StatsAnalyzer will be added '
         'automatically if any scenes have a RasterTransformer.'))

    train_chip_sz: int = Field(
        300, description='Size of training chips in pixels.')
    predict_chip_sz: int = Field(
        300, description='Size of predictions chips in pixels.')
    predict_batch_sz: int = Field(
        8, description='Batch size to use during prediction.')
    chip_nodata_threshold: Proportion = Field(
        1,
        description='Discard chips where the proportion of NODATA values is '
        'greater than or equal to this value. Might result in false positives '
        'if there are many legitimate black pixels in the chip. Use with '
        'caution.')

    analyze_uri: Optional[str] = Field(
        None,
        description=
        'URI for output of analyze. If None, will be auto-generated.')
    chip_uri: Optional[str] = Field(
        None,
        description='URI for output of chip. If None, will be auto-generated.')
    train_uri: Optional[str] = Field(
        None,
        description='URI for output of train. If None, will be auto-generated.'
    )
    predict_uri: Optional[str] = Field(
        None,
        description=
        'URI for output of predict. If None, will be auto-generated.')
    eval_uri: Optional[str] = Field(
        None,
        description='URI for output of eval. If None, will be auto-generated.')
    bundle_uri: Optional[str] = Field(
        None,
        description='URI for output of bundle. If None, will be auto-generated.'
    )
    source_bundle_uri: Optional[str] = Field(
        None,
        description='If provided, the model will be loaded from this bundle '
        'for the train stage. Useful for fine-tuning.')

    def update(self):
        super().update()

        if self.analyze_uri is None:
            self.analyze_uri = join(self.root_uri, 'analyze')
        if self.chip_uri is None:
            self.chip_uri = join(self.root_uri, 'chip')
        if self.train_uri is None:
            self.train_uri = join(self.root_uri, 'train')
        if self.predict_uri is None:
            self.predict_uri = join(self.root_uri, 'predict')
        if self.eval_uri is None:
            self.eval_uri = join(self.root_uri, 'eval')
        if self.bundle_uri is None:
            self.bundle_uri = join(self.root_uri, 'bundle')

        self.dataset.update(pipeline=self)
        self.backend.update(pipeline=self)
        if not self.evaluators:
            self.evaluators.append(self.get_default_evaluator())
        for evaluator in self.evaluators:
            evaluator.update(pipeline=self)

        self._insert_analyzers()
        for analyzer in self.analyzers:
            analyzer.update(pipeline=self)

    def get_model_bundle_uri(self):
        return join(self.bundle_uri, 'model-bundle.zip')

    def _insert_analyzers(self):
        # Inserts StatsAnalyzer if it's needed because a RasterSource has a
        # StatsTransformer, but there isn't a StatsAnalyzer in the list of Analyzers.
        has_stats_transformer = False
        for s in self.dataset.all_scenes:
            for t in s.raster_source.transformers:
                if isinstance(t, StatsTransformerConfig):
                    has_stats_transformer = True

        has_stats_analyzer = False
        for a in self.analyzers:
            if isinstance(a, StatsAnalyzerConfig):
                has_stats_analyzer = True
                break

        if has_stats_transformer and not has_stats_analyzer:
            self.analyzers.append(StatsAnalyzerConfig())

    def get_default_label_store(self, scene: SceneConfig) -> LabelStoreConfig:
        """Returns a default LabelStoreConfig to fill in any missing ones."""
        raise NotImplementedError()

    def get_default_evaluator(self) -> EvaluatorConfig:
        """Returns a default EvaluatorConfig to use if one isn't set."""
        raise NotImplementedError()
