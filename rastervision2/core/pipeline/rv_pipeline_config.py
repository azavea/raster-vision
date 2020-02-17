from os.path import join
from typing import List

from rastervision2.pipeline.pipeline_config import PipelineConfig
from rastervision2.core.data import DatasetConfig, StatsTransformerConfig
from rastervision2.core.analyzer import StatsAnalyzerConfig
from rastervision2.core.backend import BackendConfig
from rastervision2.core.evaluation import EvaluatorConfig
from rastervision2.core.analyzer import AnalyzerConfig
from rastervision2.pipeline.config import register_config


@register_config('rv_pipeline')
class RVPipelineConfig(PipelineConfig):
    dataset: DatasetConfig
    backend: BackendConfig
    evaluators: List[EvaluatorConfig] = []
    analyzers: List[AnalyzerConfig] = []

    debug: bool = False
    train_chip_sz: int = 200
    predict_chip_sz: int = 800
    predict_batch_sz: int = 8

    analyze_uri: str = None
    chip_uri: str = None
    train_uri: str = None
    predict_uri: str = None
    eval_uri: str = None
    bundle_uri: str = None

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

    def _insert_analyzers(self):
        has_stats_transformer = False
        for s in self.dataset.get_all_scenes():
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

    def get_default_label_store(self, scene):
        raise NotImplementedError()

    def get_default_evaluator(self):
        raise NotImplementedError()
