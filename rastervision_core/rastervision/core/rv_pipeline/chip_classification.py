import logging

from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.box import Box

log = logging.getLogger(__name__)


def get_train_windows(scene, chip_size, chip_nodata_threshold=1.):
    train_windows = []
    extent = scene.raster_source.get_extent()
    stride = chip_size
    windows = extent.get_windows(chip_size, stride)
    if scene.aoi_polygons:
        windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
    for window in windows:
        chip = scene.raster_source.get_chip(window)
        if (chip.sum(dim=-1) == 0).mean() < chip_nodata_threshold:
            train_windows.append(window)
    return train_windows


class ChipClassification(RVPipeline):
    def get_train_windows(self, scene):
        return get_train_windows(
            scene,
            self.config.train_chip_sz,
            chip_nodata_threshold=self.config.chip_nodata_threshold)

    def get_train_labels(self, window, scene):
        return scene.ground_truth_label_source.get_labels(window=window)
