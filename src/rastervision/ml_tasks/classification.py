from os.path import join
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from rastervision.core.ml_task import MLTask
from rastervision.evaluations.classification_evaluation import (
    ClassificationEvaluation)
from rastervision.utils.files import (
    get_local_path, upload_if_needed, make_dir)


def draw_debug_predict_image(project, class_map):
    img = project.raster_source.get_image_array()
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img, 'RGB')
    labels = project.prediction_label_store.get_all_labels()
    line_width = 4
    for cell, class_id in zip(labels.get_cells(), labels.get_class_ids()):
        cell = cell.make_eroded(line_width // 2)
        coords = cell.geojson_coordinates()
        color = class_map.get_by_id(class_id).color
        draw.line(coords, fill=color, width=line_width)
    return img


class Classification(MLTask):
    def get_train_windows(self, project, options):
        extent = project.raster_source.get_extent()
        chip_size = options.chip_size
        stride = chip_size
        windows = []
        for window in extent.get_windows(chip_size, stride):
            chip = project.raster_source.get_chip(window)
            if np.sum(chip.ravel()) > 0:
                windows.append(window)
        return windows

    def get_train_labels(self, window, project, options):
        return project.ground_truth_label_store.get_labels(window)

    def get_predict_windows(self, extent, options):
        chip_size = options.chip_size
        stride = chip_size
        return extent.get_windows(chip_size, stride)

    def get_evaluation(self):
        return ClassificationEvaluation()

    def save_debug_predict_image(self, project, debug_dir_uri):
        img = draw_debug_predict_image(project, self.class_map)
        # Saving to a jpg leads to segfault for unknown reasons.
        debug_image_uri = join(debug_dir_uri, project.id + '.png')
        with tempfile.TemporaryDirectory() as temp_dir:
            debug_image_path = get_local_path(debug_image_uri, temp_dir)
            make_dir(debug_image_path, use_dirname=True)
            img.save(debug_image_path)
            upload_if_needed(debug_image_path, debug_image_uri)
