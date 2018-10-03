from os.path import join

import numpy as np
from PIL import Image, ImageDraw

from rastervision.rv_config import RVConfig
from rastervision.core import Box
from rastervision.task import Task
from rastervision.utils.files import (get_local_path, upload_or_copy, make_dir)


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
