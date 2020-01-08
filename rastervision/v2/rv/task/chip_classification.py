import logging

import numpy as np

from rastervision.v2.rv.task.task import Task
from rastervision.v2.rv.task.chip_classification_config import (
    ChipClassificationConfig)
from rastervision.v2.rv.task import TRAIN, VALIDATION
from rastervision.v2.rv import Box, TrainingData

log = logging.getLogger(__name__)

def get_train_windows(scene, chip_size):
    extent = scene.raster_source.get_extent()
    stride = chip_size
    windows = extent.get_windows(chip_size, stride)
    if scene.aoi_polygons:
        windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
    
    train_windows = []
    for window in windows:
        chip = scene.raster_source.get_chip(window)
        if np.sum(chip.ravel()) > 0:
            train_windows.append(window)
    
    return train_windows

'''
def get_predict_windows(chip_size, extent):
    stride = chip_size
    return extent.get_windows(chip_size, stride)

def predict(self, scenes, tmp_dir):
    """Make predictions for scenes.

    The predictions are saved to the prediction_label_store in
    each scene.

    Args:
        scenes: list of Scenes
    """
    self.backend.load_model(tmp_dir)

    for scene in scenes:
        with scene.activate():
            labels = self.predict_scene(scene, tmp_dir)
            label_store = scene.prediction_label_store
            label_store.save(labels)

            if self.config.debug and self.config.predict_debug_uri:
                self.save_debug_predict_image(
                    scene, self.config.predict_debug_uri)

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
'''

class ChipClassification(Task):
    def analyze(self, split_ind=0, num_splits=1):
        pass

    def chip(self, split_ind=0, num_splits=1):
        config: ChipClassificationConfig = self.config
        backend = self.config.backend.build(config, self.tmp_dir)
        class_config = config.dataset.class_config

        def _process_scene(scene, split):
            with scene.activate():
                data = TrainingData()
                log.info('Making {} chips for scene: {}'.format(split, scene.id))
                windows = get_train_windows(scene, config.train_chip_sz)
                for window in windows:
                    chip = scene.raster_source.get_chip(window)
                    labels = scene.ground_truth_label_source.get_labels(window=window)
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
        backend.process_sceneset_results(train_results, valid_results)

    def train(self):
        backend = self.config.backend.build(self.config, self.tmp_dir)
        backend.train()

    def predict(self, split_ind=0, num_splits=1):
        pass

    def eval(self):
        pass
