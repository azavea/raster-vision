from typing import TYPE_CHECKING, List, Sequence
import logging

import numpy as np

from rastervision.core.box import Box
from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.rv_pipeline.utils import nodata_below_threshold
from rastervision.core.rv_pipeline.semantic_segmentation_config import (
    SemanticSegmentationWindowMethod)

if TYPE_CHECKING:
    from rastervision.core.backend.backend import Backend
    from rastervision.core.data import (
        ClassConfig,
        Labels,
        Scene,
        SemanticSegmentationLabelSource,
    )
    from rastervision.core.rv_pipeline.semantic_segmentation_config import (
        SemanticSegmentationConfig, SemanticSegmentationChipOptions)

log = logging.getLogger(__name__)


def get_train_windows(scene: 'Scene',
                      class_config: 'ClassConfig',
                      chip_size: int,
                      chip_options: 'SemanticSegmentationChipOptions',
                      chip_nodata_threshold: float = 1.) -> List[Box]:
    """Get training windows covering a scene.

    Args:
        scene: The scene over-which windows are to be generated.

    Returns:
        A list of windows, list(Box)
    """
    co = chip_options
    raster_source = scene.raster_source
    extent = raster_source.extent
    label_source: 'SemanticSegmentationLabelSource' = scene.label_source

    def filter_windows(windows: Sequence[Box]) -> List[Box]:
        """Filter out chips that
        (1) are outside the AOI
        (2) only consist of null labels
        (3) have NODATA proportion >= chip_nodata_threshold
        """
        total_windows = len(windows)
        if scene.aoi_polygons:
            windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
            log.info(f'AOI filtering: {len(windows)}/{total_windows} '
                     'chips accepted')

        filt_windows = []
        for w in windows:
            chip = raster_source.get_chip(w)
            nodata_below_thresh = nodata_below_threshold(
                chip, chip_nodata_threshold, nodata_val=0)

            label_arr = label_source.get_labels(w).get_label_arr(w)
            null_labels = label_arr == class_config.null_class_id

            if not np.all(null_labels) and nodata_below_thresh:
                filt_windows.append(w)
        log.info('Label and NODATA filtering: '
                 f'{len(filt_windows)}/{len(windows)} chips accepted')

        windows = filt_windows
        return windows

    def should_use_window(window: Box) -> bool:
        if co.negative_survival_prob >= 1.0:
            return True
        else:
            is_positive = False
            if co.target_class_ids is not None:
                is_positive = label_source.enough_target_pixels(
                    window, co.target_count_threshold, co.target_class_ids)
            if is_positive:
                return True
            keep_negative = np.random.sample() < co.negative_survival_prob
            return keep_negative

    if co.window_method == SemanticSegmentationWindowMethod.sliding:
        stride = co.stride or int(round(chip_size / 2))
        unfiltered_windows = extent.get_windows(chip_size, stride)
        windows = list(filter_windows(unfiltered_windows))
        if len(windows) > 0:
            a_window = windows[0]
            windows = list(filter(should_use_window, windows))
            if len(windows) == 0:
                windows = [a_window]
        elif len(windows) == 0:
            return [unfiltered_windows[0]]
    elif co.window_method == SemanticSegmentationWindowMethod.random_sample:
        windows = []
        attempts = 0

        while attempts < co.chips_per_scene:
            window = extent.make_random_square(chip_size)
            if not filter_windows([window]):
                continue

            attempts += 1
            if co.negative_survival_prob >= 1.0:
                windows.append(window)
            elif attempts == co.chips_per_scene and len(windows) == 0:
                # Ensure there is at least one window per scene.
                windows.append(window)
            elif should_use_window(window):
                windows.append(window)

    return windows


class SemanticSegmentation(RVPipeline):
    def chip(self, *args, **kwargs):
        log.info(f'Chip options: {self.config.chip_options}')
        return super().chip(*args, **kwargs)

    def get_train_windows(self, scene):
        return get_train_windows(
            scene,
            self.config.dataset.class_config,
            self.config.train_chip_sz,
            self.config.chip_options,
            chip_nodata_threshold=self.config.chip_nodata_threshold)

    def get_train_labels(self, window, scene):
        return scene.label_source.get_labels(window=window)

    def post_process_batch(self, windows, chips, labels):
        # Fill in null class for any NODATA pixels.
        null_class_id = self.config.dataset.class_config.null_class_id
        for window, chip in zip(windows, chips):
            nodata_mask = np.sum(chip, axis=2) == 0
            labels.mask_fill(window, nodata_mask, fill_value=null_class_id)

        return labels

    def predict_scene(self, scene: 'Scene', backend: 'Backend') -> 'Labels':
        cfg: 'SemanticSegmentationConfig' = self.config
        chip_sz = cfg.predict_chip_sz
        stride = cfg.predict_options.stride
        crop_sz = cfg.predict_options.crop_sz

        if stride is None:
            stride = chip_sz

        if crop_sz == 'auto':
            overlap_sz = chip_sz - stride
            if overlap_sz % 2 == 1:
                log.warn(
                    'Using crop_sz="auto" but overlap size (chip_sz minus '
                    'stride) is odd. This means that one pixel row/col will '
                    'still overlap after cropping.')
            crop_sz = overlap_sz // 2

        return backend.predict_scene(
            scene, chip_sz=chip_sz, stride=stride, crop_sz=crop_sz)
