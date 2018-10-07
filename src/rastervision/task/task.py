from abc import abstractmethod

import numpy as np
import logging

from rastervision.core.training_data import TrainingData

# TODO: DRY... same keys as in ml_backends/tf_object_detection_api.py
TRAIN = 'train'
VALIDATION = 'validation'

log = logging.getLogger(__name__)


class Task(object):
    """Functionality for a specific machine learning task.

    This should be subclassed to add a new task, such as object detection
    """

    def __init__(self, task_config, backend):
        """Construct a new Task.

        Args:
            task_config: TaskConfig
            backend: Backend
        """
        self.config = task_config
        self.backend = backend

    @abstractmethod
    def get_train_windows(self, scene):
        """Return the training windows for a Scene.

        The training windows represent the spatial extent of the training
        chips to generate.

        Args:
            scene: Scene to generate windows for

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def get_train_labels(self, window, scene):
        """Return the training labels in a window for a scene.

        Args:
            window: Box
            scene: Scene

        Returns:
            Labels that lie within window
        """
        pass

    @abstractmethod
    def post_process_predictions(self, labels, scene):
        """Runs a post-processing step on labels at end of prediction.

        Returns:
            Labels
        """
        pass

    @abstractmethod
    def get_predict_windows(self, extent):
        """Return windows to compute predictions for.

        Args:
            extent: Box representing extent of RasterSource

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def save_debug_predict_image(self, scene, debug_dir_uri):
        """Save a debug image of predictions.

        This writes to debug_dir_uri/<scene.id>.jpg.
        """
        pass

    def make_chips(self, train_scenes, validation_scenes, augmentors, tmp_dir):
        """Make training chips.

        Convert Scenes with a ground_truth_label_store into training
        chips in MLBackend-specific format, and write to URI specified in
        options.

        Args:
            train_scenes: list of Scene
            validation_scenes: list of Scene
                (that is disjoint from train_scenes)
            augmentors: Augmentors used to augment training data
        """

        def _process_scene(scene, type_, augment):
            data = TrainingData()
            log.info('Making {} chips for scene: {}'.format(type_, scene.id))
            windows = self.get_train_windows(scene)
            for window in windows:
                chip = scene.raster_source.get_chip(window)
                labels = self.get_train_labels(window, scene)
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

        # TODO: parallel processing!
        processed_training_results = _process_scenes(
            train_scenes, TRAIN, augment=True)
        processed_validation_results = _process_scenes(
            validation_scenes, VALIDATION, augment=False)

        self.backend.process_sceneset_results(
            processed_training_results, processed_validation_results, tmp_dir)

    def train(self, tmp_dir):
        """Train a model.
        """
        self.backend.train(tmp_dir)

    def predict(self, scenes, tmp_dir):
        """Make predictions for scenes.

        The predictions are saved to the prediction_label_store in
        each scene.

        Args:
            scenes: list of Scenes
        """
        self.backend.load_model(tmp_dir)

        for scene in scenes:
            labels = self.predict_scene(scene, tmp_dir)
            label_store = scene.prediction_label_store
            label_store.save(labels)

            if self.config.debug and self.config.predict_debug_uri:
                self.save_debug_predict_image(scene,
                                              self.config.predict_debug_uri)

    def predict_scene(self, scene, tmp_dir):
        """Predict on a single scene, and return the labels."""
        log.info('Making predictions for scene')
        raster_source = scene.raster_source
        label_store = scene.prediction_label_store
        labels = label_store.empty_labels()

        windows = self.get_predict_windows(raster_source.get_extent())

        def predict_batch(predict_chips, predict_windows):
            nonlocal labels
            new_labels = self.backend.predict(
                np.array(predict_chips), predict_windows, tmp_dir)
            labels += new_labels
            print('.' * len(predict_chips), end='', flush=True)

        batch_chips, batch_windows = [], []
        for window in windows:
            chip = raster_source.get_chip(window)
            if np.any(chip):
                batch_chips.append(chip)
                batch_windows.append(window)

            # Predict on batch
            if len(batch_chips) >= self.config.predict_batch_size:
                predict_batch(batch_chips, batch_windows)
                batch_chips, batch_windows = [], []
        print()

        # Predict on remaining batch
        if len(batch_chips) > 0:
            predict_batch(batch_chips, batch_windows)

        return self.post_process_predictions(labels, scene)
