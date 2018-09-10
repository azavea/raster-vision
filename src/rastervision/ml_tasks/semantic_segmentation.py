import numpy as np

from typing import List

from rastervision.core.box import Box
from rastervision.core.ml_task import MLTask
from rastervision.core.scene import Scene
from rastervision.evaluations.segmentation_evaluation import (
    SegmentationEvaluation)


class SemanticSegmentation(MLTask):
    """MLTask-derived type that implements the semantic segmentation task.

    """

    def get_train_windows(self, scene: Scene, options) -> List[Box]:
        """Get training windows covering a scene.

        Args:
             scene: The scene over-which windows are to be generated.
             options: Options passed through from the
                  `make_training_chips` section of the workflow
                  configuration file.

        Returns:
             A list of windows, list(Box)

        """
        seg_options = options.segmentation_options
        raster_source = scene.raster_source
        extent = raster_source.get_extent()
        label_store = scene.ground_truth_label_store
        chip_size = options.chip_size
        prob = seg_options.negative_survival_probability
        ioa_threshold = seg_options.ioa_threshold
        target_classes = seg_options.target_classes
        number_of_chips = seg_options.number_of_chips
        if len(target_classes) == 0:
            target_classes = [1]

        windows = []
        attempts = 0
        while (attempts < number_of_chips):
            attempts = attempts + 1
            candidate_window = extent.make_random_square(chip_size)
            if (prob >= 1.0):
                windows.append(candidate_window)
            elif attempts == number_of_chips and len(windows) == 0:
                windows.append(candidate_window)
            else:
                good = label_store.enough_target_pixels(
                    candidate_window, ioa_threshold, target_classes)
                if good or (np.random.rand() < prob):
                    windows.append(candidate_window)

        return windows

    def get_train_labels(self, window: Box, scene: Scene,
                         options) -> np.ndarray:
        """Get the training labels for the given window in the given scene.

        Args:
             window: The window over-which the labels are to be
                  retrieved.
             scene: The scene from-which the window of labels is to be
                  extracted.
             options: Options passed through from the
                  `make_training_chips` section of the workflow
                  configuration file.

        Returns:
             An appropriately-shaped 2d np.ndarray with the labels
             encoded as packed pixels.

        """
        label_store = scene.ground_truth_label_store
        return label_store.get_labels(window)

    def get_predict_windows(self, extent: Box, options) -> List[Box]:
        """Get windows over-which predictions will be calculated.

        Args:
             extent: The overall extent of the area.
             options: Options from the prediction section of the
                  workflow configuration file.

        Returns:
             An sequence of windows.

        """
        chip_size = options.chip_size
        return extent.get_windows(chip_size, chip_size)

    def post_process_predictions(self, labels: None, options) -> None:
        """Post-process predictions.

        Is a nop for this backend.
        """
        return None

    def get_evaluation(self) -> SegmentationEvaluation:
        """Return a segmentation  evaulation object."""
        return SegmentationEvaluation()
