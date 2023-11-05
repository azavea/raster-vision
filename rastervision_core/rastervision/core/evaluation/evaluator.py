from typing import TYPE_CHECKING, Any, Iterable
from abc import (ABC, abstractmethod)

if TYPE_CHECKING:
    from rastervision.core.data import Scene, Labels


class Evaluator(ABC):
    """Evaluates predictions for a set of scenes."""

    @abstractmethod
    def process(self, scenes: Iterable['Scene']) -> None:
        """Evaluate all given scenes and save the evaluations.

        Args:
            scenes (Iterable[Scene]): Scenes to evaluate.
        """

    @abstractmethod
    def evaluate_scene(self, scene: 'Scene') -> Any:
        """Evaluate predictions from a scene's labels store.

        The predictions are evalated against ground truth labels from the
        scene's label source.

        Args:
            scene (Scene): A scene with a label source and a label store.

        Returns:
            ClassificationEvaluation: The evaluation.
        """

    @abstractmethod
    def evaluate_predictions(self, ground_truth: 'Labels',
                             predictions: 'Labels') -> Any:
        """Evaluate predictions against ground truth.

        Args:
            ground_truth (Labels): Ground truth labels.
            predictions (Labels): Predictions.

        Returns:
            Any: The evaluation.
        """
