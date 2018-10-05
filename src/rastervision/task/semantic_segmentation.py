import numpy as np

from typing import List

from .task import Task
from rastervision.core.box import Box
from rastervision.data.scene import Scene


def get_random_sample_train_windows(label_store, chip_size, class_map, extent,
                                    chip_options, filter_windows):
    prob = chip_options.negative_survival_probability
    target_count_threshold = chip_options.target_count_threshold
    target_classes = chip_options.target_classes
    chips_per_scene = chip_options.chips_per_scene

    if not target_classes:
        all_class_ids = [item.id for item in class_map.get_items()]
        target_classes = all_class_ids

    windows = []
    attempts = 0
    while (attempts < chips_per_scene):
        candidate_window = extent.make_random_square(chip_size)
        if not filter_windows([candidate_window]):
            continue
        attempts = attempts + 1

        if (prob >= 1.0):
            windows.append(candidate_window)
        elif attempts == chips_per_scene and len(windows) == 0:
            windows.append(candidate_window)
        else:
            good = label_store.enough_target_pixels(
                candidate_window, target_count_threshold, target_classes)
            if good or (np.random.rand() < prob):
                windows.append(candidate_window)

    return windows


class SemanticSegmentation(Task):
    """Task-derived type that implements the semantic segmentation task."""

    def get_train_windows(self, scene: Scene) -> List[Box]:
        """Get training windows covering a scene.

        Args:
             scene: The scene over-which windows are to be generated.

        Returns:
             A list of windows, list(Box)

        """

        def filter_windows(windows):
            if scene.aoi_polygons:
                windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
            return windows

        raster_source = scene.raster_source
        extent = raster_source.get_extent()
        label_store = scene.ground_truth_label_source
        chip_size = self.config.chip_size

        chip_options = self.config.chip_options

        if chip_options.window_method == 'random_sample':
            return get_random_sample_train_windows(
                label_store, chip_size, self.config.class_map, extent,
                chip_options, filter_windows)
        elif chip_options.window_method == 'sliding':
            stride = chip_options.stride
            if stride is None:
                stride = chip_size / 2

            return list(
                filter_windows((extent.get_windows(chip_size, stride))))

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
        return labels

    def save_debug_predict_image(self, scene, debug_dir_uri):
        # TODO implement this
        pass
