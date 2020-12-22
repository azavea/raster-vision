import logging
from typing import List, Sequence

import numpy as np

from rastervision.core.box import Box
from rastervision.core.data import ClassConfig, Scene
from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.rv_pipeline.utils import (fill_no_data,
                                                 nodata_below_threshold)
from rastervision.core.rv_pipeline.semantic_segmentation_config import (
    SemanticSegmentationWindowMethod, SemanticSegmentationChipOptions)

log = logging.getLogger(__name__)


def get_train_windows(scene: Scene,
                      class_config: ClassConfig,
                      chip_size: int,
                      chip_options: SemanticSegmentationChipOptions,
                      chip_nodata_threshold: float = 1.) -> List[Box]:
    """Get training windows covering a scene.

    Args:
        scene: The scene over-which windows are to be generated.

    Returns:
        A list of windows, list(Box)
    """
    co = chip_options
    raster_source = scene.raster_source
    extent = raster_source.get_extent()
    label_source = scene.ground_truth_label_source

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
            null_labels = label_arr == class_config.get_null_class_id()

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
            target_class_ids = co.target_class_ids or list(
                range(len(class_config)))
            is_pos = label_source.enough_target_pixels(
                window, co.target_count_threshold, target_class_ids)
            should_use = is_pos or (np.random.rand() <
                                    co.negative_survival_prob)
            return should_use

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
    def __init__(self, config: 'RVPipelineConfig', tmp_dir: str):
        super().__init__(config, tmp_dir)
        if self.config.dataset.img_channels is None:
            self.config.dataset.img_channels = self.get_img_channels()

            self.config.dataset.update()
            self.config.dataset.validate_config()

            self.config.update()
            self.config.validate_config()

    def get_img_channels(self):
        ''' Determine img_channels from the first training scene. '''
        class_config = self.config.dataset.class_config
        scene_cfg = self.config.dataset.train_scenes[0]
        scene = scene_cfg.build(
            class_config, self.tmp_dir, use_transformers=False)
        with scene.activate():
            img_channels = scene.raster_source.num_channels
        return img_channels

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
        return scene.ground_truth_label_source.get_labels(window=window)

    def post_process_sample(self, sample):
        # Use null label for each pixel with NODATA.
        img = sample.chip
        label_arr = sample.labels.get_label_arr(sample.window)
        null_class_id = self.config.dataset.class_config.get_null_class_id()
        sample.chip = fill_no_data(img, label_arr, null_class_id)
        return sample

    def post_process_batch(self, windows, chips, labels):
        # Fill in null class for any NODATA pixels.
        null_class_id = self.config.dataset.class_config.get_null_class_id()
        for window, chip in zip(windows, chips):
            nodata_mask = np.sum(chip, axis=2) == 0
            labels.mask_fill(window, nodata_mask, fill_value=null_class_id)

        return labels

    def get_predict_windows(self, extent: Box) -> List[Box]:
        chip_sz = self.config.predict_chip_sz
        stride = self.config.predict_options.stride
        if stride is None:
            stride = chip_sz
        return extent.get_windows(chip_sz, stride)
