import logging

import numpy as np

from rastervision.v2.core.pipeline import Pipeline
from rastervision.v2.rv import Box, TrainingData
from rastervision.v2.rv.task import TRAIN, VALIDATION

log = logging.getLogger(__name__)

class Task(Pipeline):
    commands = ['analyze', 'chip', 'train', 'predict', 'eval']
    split_commands = ['analyze', 'chip', 'predict']
    gpu_commands = ['train', 'predict']

    def analyze(self, split_ind=0, num_splits=1):
        pass

    def get_train_windows(self, scene):
        """Return the training windows for a Scene.

        The training windows represent the spatial extent of the training
        chips to generate.

        Args:
            scene: Scene to generate windows for

        Returns:
            list of Boxes
        """
        raise NotImplementedError()

    def get_train_labels(self, window, scene):
        """Return the training labels in a window for a scene.

        Args:
            window: Box
            scene: Scene

        Returns:
            Labels that lie within window
        """
        raise NotImplementedError()

    def chip(self, split_ind=0, num_splits=1):
        # TODO handle split
        config = self.config
        backend = self.config.backend.build(config, self.tmp_dir)
        class_config = config.dataset.class_config

        def _process_scene(scene, split):
            with scene.activate():
                data = TrainingData()
                log.info('Making {} chips for scene: {}'.format(split, scene.id))
                windows = self.get_train_windows(scene)
                for window in windows:
                    chip = scene.raster_source.get_chip(window)
                    labels = self.get_train_labels(window, scene)
                    data.append(chip, window, labels)
                # Shuffle data so the first N samples which are displayed in
                # Tensorboard are more diverse.
                data.shuffle()
                return backend.process_scene_data(scene, data)

        def _process_scenes(scenes, split):
            return [_process_scene(s.build(class_config, self.tmp_dir), split)
                    for s in config.dataset.train_scenes]

        train_results = _process_scenes(config.dataset.train_scenes, TRAIN)
        valid_results = _process_scenes(
            config.dataset.validation_scenes, VALIDATION)
        backend.process_sceneset_results(
            train_results, valid_results)

    def train(self):
        backend = self.config.backend.build(self.config, self.tmp_dir)
        backend.train()

    def post_process_predictions(self, labels, scene):
        """Runs a post-processing step on labels at end of prediction.

        Returns:
            Labels
        """
        raise NotImplementedError()

    def get_predict_windows(self, extent):
        """Return windows to compute predictions for.

        Args:
            extent: Box representing extent of RasterSource

        Returns:
            list of Boxes
        """
        raise NotImplementedError()

    def predict(self, split_ind=0, num_splits=1):
        backend = self.config.backend.build(self.config, self.tmp_dir)
        backend.load_model()

        def _predict(scenes):
            for scene in scenes:
                with scene.activate():
                    labels = self.predict_scene(scene, backend)
                    label_store = scene.prediction_label_store
                    label_store.save(labels)

        class_config = self.config.dataset.class_config
        _predict([s.build(class_config, self.tmp_dir)
                  for s in self.config.dataset.validation_scenes])
        if self.config.dataset.test_scenes:
            _predict([s.build(class_config, self.tmp_dir)
                    for s in self.config.dataset.test_scenes])

    def predict_scene(self, scene, backend):
        """Predict on a single scene, and return the labels."""
        log.info('Making predictions for scene')
        raster_source = scene.raster_source
        label_store = scene.prediction_label_store
        labels = label_store.empty_labels()

        windows = self.get_predict_windows(raster_source.get_extent())

        def predict_batch(predict_chips, predict_windows):
            nonlocal labels
            new_labels = backend.predict(
                np.array(predict_chips), predict_windows)
            labels += new_labels
            print('.' * len(predict_chips), end='', flush=True)

        batch_chips, batch_windows = [], []
        for window in windows:
            chip = raster_source.get_chip(window)
            if np.any(chip):
                batch_chips.append(chip)
                batch_windows.append(window)

            # Predict on batch
            if len(batch_chips) >= self.config.predict_batch_sz:
                predict_batch(batch_chips, batch_windows)
                batch_chips, batch_windows = [], []
        print()

        # Predict on remaining batch
        if len(batch_chips) > 0:
            predict_batch(batch_chips, batch_windows)

        return self.post_process_predictions(labels, scene)

    def eval(self):
        pass
