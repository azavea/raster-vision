import numpy as np

from typing import List

from .task import Task
from rastervision.core.box import Box
from rastervision.data.scene import Scene


class SemanticSegmentation(Task):
    """Task-derived type that implements the semantic segmentation task."""

    def get_train_windows(self, scene: Scene) -> List[Box]:
        """Get training windows covering a scene.

        Args:
             scene: The scene over-which windows are to be generated.

        Returns:
             A list of windows, list(Box)

        """
        raster_source = scene.raster_source
        extent = raster_source.get_extent()
        label_store = scene.ground_truth_label_source
        chip_size = self.config.chip_size

        chip_options = self.config.chip_options
        prob = chip_options.negative_survival_probability
        target_count_threshold = chip_options.target_count_threshold
        target_classes = chip_options.target_classes
        number_of_chips = chip_options.number_of_chips

        if not target_classes:
            all_class_ids = [
                item.id for item in self.config.class_map.get_items()
            ]
            target_classes = all_class_ids

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
                    candidate_window, target_count_threshold, target_classes)
                if good or (np.random.rand() < prob):
                    windows.append(candidate_window)

        return windows

    def get_train_labels(self, window: Box, scene: Scene) -> np.ndarray:
        """Get the training labels for the given window in the given scene.

        Args:
             window: The window over-which the labels are to be
                  retrieved.
             scene: The scene from-which the window of labels is to be
                  extracted.

        Returns:
             An appropriately-shaped 2d np.ndarray with the labels
             encoded as packed pixels.

        """
        label_store = scene.ground_truth_label_source
        return label_store.get_labels(window)

    def get_predict_windows(self, extent: Box) -> List[Box]:
        """Get windows over-which predictions will be calculated.

        Args:
             extent: The overall extent of the area.

        Returns:
             An sequence of windows.

        """
        chip_size = self.config.chip_size
        return extent.get_windows(chip_size, chip_size)

    def post_process_predictions(self, labels, scene):
        """Post-process predictions.

        Cut off "junk" predictions made off the edge of scene.raster_source

        Returns: SemanticSegmentationLabels
        """
        extent = scene.raster_source.get_extent()
        return labels.get_clipped_labels(extent)

    def save_debug_predict_image(self, scene, debug_dir_uri):
        # TODO implement this
        pass
