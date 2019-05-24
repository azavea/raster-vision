from typing import List
import logging

import numpy as np

from .task import Task
from rastervision.core.box import Box
from rastervision.data.scene import Scene
from rastervision.data.label import SemanticSegmentationLabels
from rastervision.core.training_data import TrainingData

TRAIN = 'train'
VALIDATION = 'validation'

log = logging.getLogger(__name__)


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
        raster_source = scene.raster_source
        extent = raster_source.get_extent()
        label_source = scene.ground_truth_label_source
        chip_size = self.config.chip_size
        chip_options = self.config.chip_options

        def filter_windows(windows):
            if scene.aoi_polygons:
                windows = Box.filter_by_aoi(windows, scene.aoi_polygons)

            if 0 in self.config.class_map.get_keys():
                filt_windows = []
                for w in windows:
                    label_arr = label_source.get_labels(w).get_label_arr(w)
                    ignore_inds = label_arr.ravel() == 0
                    if np.all(ignore_inds):
                        pass
                    else:
                        filt_windows.append(w)
                windows = filt_windows
            return windows

        if chip_options.window_method == 'random_sample':
            return get_random_sample_train_windows(
                label_source, chip_size, self.config.class_map, extent,
                chip_options, filter_windows)
        elif chip_options.window_method == 'sliding':
            stride = chip_options.stride
            if stride is None:
                stride = chip_size / 2
            stride = int(round(stride))

            return list(
                filter_windows((extent.get_windows(chip_size, stride))))

    def make_chips(self, train_scenes, validation_scenes, augmentors, tmp_dir):
        """Make training chips.

        Convert Scenes with a ground_truth_label_store into training
        chips in MLBackend-specific format, and write to URI specified in
        options.

        Args:
            train_scenes: list of Scenes
            validation_scenes: list of Scenes
                (that is disjoint from train_scenes)
            augmentors: Augmentors used to augment training data
        """

        def _process_scene(scene, type_, augment):
            with scene.activate():
                data = TrainingData()
                log.info('Making {} chips for scene: {}'.format(
                    type_, scene.id))
                windows = self.get_train_windows(scene)
                for window in windows:
                    chip = scene.raster_source.get_chip(window)
                    labels = self.get_train_labels(window, scene)

                    # If chip has ignore labels, fill in those pixels with
                    # nodata.
                    label_arr = labels.get_label_arr(window)
                    zero_inds = label_arr.ravel() == 0
                    chip_shape = chip.shape
                    if np.any(zero_inds):
                        chip = np.reshape(chip, (-1, chip.shape[2]))
                        chip[zero_inds, :] = 0
                        chip = np.reshape(chip, chip_shape)

                    data.append(chip, window, labels)
                # Shuffle data so the first N samples which are displayed in
                # Tensorboard are more diverse.
                data.shuffle()

                # Process augmentation
                if augment:
                    for augmentor in augmentors:
                        data = augmentor.process(data, tmp_dir)

                return self.backend.process_scene_data(scene, data, tmp_dir)

        def _process_scenes(scenes, type_, augment):
            return [_process_scene(scene, type_, augment) for scene in scenes]

        processed_training_results = _process_scenes(
            train_scenes, TRAIN, augment=True)
        processed_validation_results = _process_scenes(
            validation_scenes, VALIDATION, augment=False)

        self.backend.process_sceneset_results(
            processed_training_results, processed_validation_results, tmp_dir)

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
        chip_size = self.config.predict_chip_size
        return extent.get_windows(chip_size, chip_size)

    def post_process_predictions(self, labels, scene):
        return labels

    def save_debug_predict_image(self, scene, debug_dir_uri):
        # TODO implement this
        pass

    def predict_scene(self, scene, tmp_dir):
        """Predict on a single scene, and return the labels."""
        log.info('Making predictions for scene')
        raster_source = scene.raster_source
        windows = self.get_predict_windows(raster_source.get_extent())

        def label_fn(window):
            chip = raster_source.get_chip(window)
            labels = self.backend.predict([chip], [window], tmp_dir)
            label_arr = labels.get_label_arr(window)

            # Set NODATA pixels in imagery to predicted value of 0 (ie. ignore)
            label_arr[np.sum(chip, axis=2) == 0] = 0

            print('.', end='', flush=True)
            return label_arr

        return SemanticSegmentationLabels(windows, label_fn)
