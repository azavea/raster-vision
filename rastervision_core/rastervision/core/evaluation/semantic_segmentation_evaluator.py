from typing import TYPE_CHECKING
import logging

from rastervision.core.evaluation import (ClassificationEvaluator,
                                          SemanticSegmentationEvaluation)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rastervision.core.data import Scene


class SemanticSegmentationEvaluator(ClassificationEvaluator):
    """Evaluates predictions for a set of scenes."""

    def create_evaluation(self) -> SemanticSegmentationEvaluation:
        return SemanticSegmentationEvaluation(self.class_config)

    def evaluate_scene(self, scene: 'Scene') -> SemanticSegmentationEvaluation:
        """Override to pass null_class_id to filter_by_aoi()."""
        null_class_id = self.class_config.null_class_id
        ground_truth = scene.label_source.get_labels()
        predictions = scene.label_store.get_labels()

        if scene.aoi_polygons:
            ground_truth = ground_truth.filter_by_aoi(scene.aoi_polygons,
                                                      null_class_id)
            predictions = predictions.filter_by_aoi(scene.aoi_polygons,
                                                    null_class_id)
        evaluation = self.evaluate_predictions(ground_truth, predictions)
        return evaluation
