from typing import TYPE_CHECKING, Iterable, Optional
from abc import (abstractmethod)
import logging

from rastervision.core.evaluation import Evaluator
from rastervision.core.data import Labels

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.core.data import Scene, ClassConfig
    from rastervision.core.evaluation import ClassificationEvaluation


class ClassificationEvaluator(Evaluator):
    """Evaluates predictions for a set of scenes."""

    def __init__(self,
                 class_config: 'ClassConfig',
                 output_uri: Optional[str] = None):
        self.class_config = class_config
        self.output_uri = output_uri

    @abstractmethod
    def create_evaluation(self) -> 'ClassificationEvaluation':
        pass

    def process(self, scenes: Iterable['Scene'],
                tmp_dir: Optional[str] = None) -> None:
        if self.output_uri is not None:
            evaluation_global = self.create_evaluation()
            for scene in scenes:
                log.info(f'Computing evaluation for scene {scene.id}...')
                evaluation = self.evaluate_scene(scene)
                evaluation_global.merge(evaluation, scene_id=scene.id)
            evaluation_global.save(self.output_uri)

    def evaluate_scene(self, scene: 'Scene') -> 'ClassificationEvaluation':
        ground_truth = scene.label_source.get_labels()
        predictions = scene.label_store.get_labels()

        if scene.aoi_polygons:
            ground_truth = ground_truth.filter_by_aoi(scene.aoi_polygons)
            predictions = predictions.filter_by_aoi(scene.aoi_polygons)

        evaluation = self.evaluate_predictions(ground_truth, predictions)
        return evaluation

    def evaluate_predictions(
            self, ground_truth: 'Labels',
            predictions: 'Labels') -> 'ClassificationEvaluation':
        evaluation = self.create_evaluation()
        evaluation.compute(ground_truth, predictions)
        return evaluation
