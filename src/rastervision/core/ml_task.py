from abc import abstractmethod

from rastervision.core.training_data import TrainingData
from rastervision.core.predict_package import save_predict_package

import numpy as np

# TODO: DRY... same keys as in ml_backends/tf_object_detection_aip.py
TRAIN = 'train'
VALIDATION = 'validation'


class MLTask(object):
    """Functionality for a specific machine learning task.

    This should be subclassed to add a new task, such as object detection
    """

    def __init__(self, backend, class_map):
        """Construct a new MLTask.

        Args:
            backend: MLBackend
            class_map: ClassMap
        """
        self.backend = backend
        self.class_map = class_map

    @abstractmethod
    def get_train_windows(self, scene, options):
        """Return the training windows for a Scene.

        The training windows represent the spatial extent of the training
        chips to generate.

        Args:
            scene: Scene to generate windows for
            options: TrainConfig.Options

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def get_train_labels(self, window, scene, options):
        """Return the training labels in a window for a scene.

        Args:
            window: Box
            scene: Scene
            options: TrainConfig.Options

        Returns:
            Labels that lie within window
        """
        pass

    @abstractmethod
    def post_process_predictions(self, labels, options):
        """Runs a post-processing step on labels at end of prediction.

        Args:
            options: PredictConfig.Options

        Returns:
            Labels
        """
        pass

    @abstractmethod
    def get_predict_windows(self, extent, options):
        """Return windows to compute predictions for.

        Args:
            extent: Box representing extent of RasterSource
            options: PredictConfig.Options

        Returns:
            list of Boxes
        """
        pass

    @abstractmethod
    def get_evaluation(self):
        """Return empty Evaluation of appropriate type.

        This functions as a factory.
        """
        pass

    @abstractmethod
    def save_debug_predict_image(self, scene, debug_dir_uri):
        """Save a debug image of predictions.

        This writes to debug_dir_uri/<scene.id>.jpg.
        """
        pass

    def get_class_map(self):
        return self.class_map

    def transform_training_chip(self, window, chip, labels, options):
        """Randomly transform a training chip for purposes of data aug.

        This default implementation is a no-op.

        Args:
            window: Box
            chip: numpy array of shape [chip_size, chip_size, nb_channels]
            options: MakeTrainingChipsConfig.Options

        Returns: transformed (chip, labels)
        """
        return chip, labels

    def make_training_chips(self, train_scenes, validation_scenes, options):
        """Make training chips.

        Convert Scenes with a ground_truth_label_store into training
        chips in MLBackend-specific format, and write to URI specified in
        options.

        Args:
            train_scenes: list of Scene
            validation_scenes: list of Scene
                (that is disjoint from train_scenes)
            options: MakeTrainingChipsConfig.Options
        """

        def _process_scene(scene, type_):
            data = TrainingData()
            print(
                'Making {} chips for scene: {}'.format(type_, scene.id),
                end='',
                flush=True)
            windows = self.get_train_windows(scene, options)
            for window in windows:
                chip = scene.raster_source.get_chip(window)
                labels = self.get_train_labels(window, scene, options)
                # Only do data augmentation on training set.
                if type_ == TRAIN:
                    chip, labels = self.transform_training_chip(
                        window, chip, labels, options)
                data.append(chip, window, labels)
                print('.', end='', flush=True)
            print()

            # Shuffle data so the first N samples which are displayed in
            # Tensorboard are more diverse.
            data.shuffle()
            # TODO load and delete scene data as needed to avoid
            # running out of disk space
            return self.backend.process_scene_data(scene, data, self.class_map,
                                                   options)

        def _process_scenes(scenes, type_):
            return [_process_scene(scene, type_) for scene in scenes]

        # TODO: parallel processing!
        processed_training_results = _process_scenes(train_scenes, TRAIN)
        processed_validation_results = _process_scenes(validation_scenes,
                                                       VALIDATION)
        self.backend.process_sceneset_results(processed_training_results,
                                              processed_validation_results,
                                              self.class_map, options)

    def train(self, options):
        """Train a model.

        Args:
            options: TrainConfig.options
        """
        self.backend.train(self.class_map, options)

    def predict(self, scenes, config):
        """Make predictions for scenes.

        The predictions are saved to the prediction_label_store in
        each scene.

        Args:
            scenes: list of Scenes
            config: PredictConfig
        """
        options = config.options
        for scene in scenes:
            print('Making predictions for scene', end='', flush=True)
            raster_source = scene.raster_source
            label_store = scene.prediction_label_store
            label_store.clear()

            windows = self.get_predict_windows(raster_source.get_extent(),
                                               options)
            for window in windows:
                chip = raster_source.get_chip(window)

                # Only predict on chips containing at least one NODATA value.
                if np.any(chip):
                    labels = self.backend.predict(chip, window, options)
                    label_store.extend(labels)
                    print('.', end='', flush=True)
            print()

            labels = self.post_process_predictions(label_store.get_labels(),
                                                   options)
            label_store.set_labels(labels)
            label_store.save()

            if (options.debug and options.debug_uri
                    and self.class_map.has_all_colors()):
                self.save_debug_predict_image(scene, options.debug_uri)

            if options.prediction_package_uri:
                save_predict_package(config)

    def eval(self, scenes, options):
        """Evaluate predictions against ground truth in scenes.

        Writes output to URI in options.

        Args:
            scenes: list of Scenes that contain both
                ground_truth_label_store and prediction_label_store
            options: EvalConfig.Options
        """
        evaluation = self.get_evaluation()
        for scene in scenes:
            print('Computing evaluation for scene...')
            ground_truth = scene.ground_truth_label_store
            predictions = scene.prediction_label_store

            scene_evaluation = self.get_evaluation()
            scene_evaluation.compute(self.class_map, ground_truth, predictions)
            evaluation.merge(scene_evaluation)
        evaluation.save(options.output_uri)
