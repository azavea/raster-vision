from os.path import join

import numpy as np
from PIL import Image, ImageDraw
import logging

from rastervision.rv_config import RVConfig
from rastervision.core import Box
from rastervision.task import Task
from rastervision.utils.files import (get_local_path, upload_or_copy, make_dir)
from rastervision.core.training_data import TrainingData

log = logging.getLogger(__name__)

# TODO: DRY... same keys as in ml_backends/tf_object_detection_api.py
TRAIN = 'train'
VALIDATION = 'validation'


def draw_debug_predict_image(scene, class_map):
    img = scene.raster_source.get_image_array()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, 'RGB')
    labels = scene.prediction_label_store.get_labels()
    line_width = 4
    default_colors = [
        'red', 'orange', 'yellow', 'green', 'brown', 'pink', 'purple'
    ]
    for cell, class_id in zip(labels.get_cells(), labels.get_class_ids()):
        cell = cell.make_eroded(line_width // 2)
        coords = cell.geojson_coordinates()
        color = class_map.get_by_id(class_id).color
        if color is None:
            color = default_colors[(class_id - 1) % len(default_colors)]
        draw.line(coords, fill=color, width=line_width)
    return img


def correct_imbalance(data, max_imbalance, class_map):
    # Assume a minimum count of 1 for each class to handle the case of
    # there being no instances of a class.
    class_to_count = {}
    for class_id in class_map.get_keys():
        class_to_count[class_id] = 1
    class_to_data = {}
    for chip_idx, (chip, window, labels) in enumerate(data):
        class_id = labels.get_cell_class_id(window)
        # If a chip is not associated with a class, don't
        # use it in training data.
        if class_id is None:
            continue

        class_to_count[class_id] += 1
        class_data = class_to_data.get(class_id, [])
        class_data.append((chip, window, labels))
        class_to_data[class_id] = class_data

    min_count = min(list(class_to_count.values()))
    max_count = max(list(class_to_count.values()))
    max_allowable = int(max_imbalance * min_count)
    if max_count > max_allowable:
        for class_id in class_to_data.keys():
            desired_count = min(class_to_count[class_id], max_allowable)
            class_to_data[class_id] = \
                class_to_data[class_id][0:desired_count]

    data = TrainingData()
    for class_id, class_data in class_to_data.items():
        for chip, window, labels in class_data:
            data.append(chip, window, labels)

    return data
    

class ChipClassification(Task):
    def get_train_windows(self, scene):
        result = []
        extent = scene.raster_source.get_extent()
        chip_size = self.config.chip_size
        stride = chip_size
        windows = extent.get_windows(chip_size, stride)
        if scene.aoi_polygons:
            windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
        for window in windows:
            chip = scene.raster_source.get_chip(window)
            if np.sum(chip.ravel()) > 0:
                result.append(window)
        return result

    def get_train_labels(self, window, scene):
        return scene.ground_truth_label_source.get_labels(window=window)

    def post_process_predictions(self, labels, scene):
        return labels

    def get_predict_windows(self, extent):
        chip_size = self.config.chip_size
        stride = chip_size
        return extent.get_windows(chip_size, stride)

    def save_debug_predict_image(self, scene, debug_dir_uri):
        img = draw_debug_predict_image(scene, self.config.class_map)
        # Saving to a jpg leads to segfault for unknown reasons.
        debug_image_uri = join(debug_dir_uri, scene.id + '.png')
        with RVConfig.get_tmp_dir() as temp_dir:
            debug_image_path = get_local_path(debug_image_uri, temp_dir)
            make_dir(debug_image_path, use_dirname=True)
            img.save(debug_image_path)
            upload_or_copy(debug_image_path, debug_image_uri)

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
            with scene.activate():
                data = TrainingData()
                log.info('Making {} chips for scene: {}'.format(
                    type_, scene.id))
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

                max_imbalance = self.config.max_imbalance
                if max_imbalance != 0:
                    data = correct_imbalance(data, max_imbalance, self.config.class_map)

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
