import logging

import numpy as np

from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.box import Box

log = logging.getLogger(__name__)


def get_train_windows(scene, chip_size):
    train_windows = []
    extent = scene.raster_source.get_extent()
    stride = chip_size
    windows = extent.get_windows(chip_size, stride)
    if scene.aoi_polygons:
        windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
    for window in windows:
        chip = scene.raster_source.get_chip(window)
        if np.sum(chip.ravel()) > 0:
            train_windows.append(window)
    return train_windows


class ChipClassification(RVPipeline):
    def get_train_windows(self, scene):
        return get_train_windows(scene, self.config.train_chip_sz)

    def get_train_labels(self, window, scene):
        return scene.ground_truth_label_source.get_labels(window=window)
