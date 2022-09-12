from typing import List
import logging

from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.rv_pipeline.utils import nodata_below_threshold
from rastervision.core.box import Box
from rastervision.core.data import Scene

log = logging.getLogger(__name__)


def get_train_windows(scene: Scene,
                      chip_size: int,
                      chip_nodata_threshold: float = 1.) -> List[Box]:
    train_windows = []
    extent = scene.raster_source.extent
    stride = chip_size
    windows = extent.get_windows(chip_size, stride)

    total_windows = len(windows)
    if scene.aoi_polygons:
        windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
        log.info(f'AOI filtering: {len(windows)}/{total_windows} '
                 'chips accepted')
    for window in windows:
        chip = scene.raster_source.get_chip(window)
        if nodata_below_threshold(chip, chip_nodata_threshold, nodata_val=0):
            train_windows.append(window)
    log.info('NODATA filtering: '
             f'{len(train_windows)}/{len(windows)} chips accepted')
    return train_windows


class ChipClassification(RVPipeline):
    def get_train_windows(self, scene: Scene) -> List[Box]:
        return get_train_windows(
            scene,
            self.config.train_chip_sz,
            chip_nodata_threshold=self.config.chip_nodata_threshold)

    def get_train_labels(self, window: Box, scene: Scene):
        return scene.label_source.get_labels(window=window)
