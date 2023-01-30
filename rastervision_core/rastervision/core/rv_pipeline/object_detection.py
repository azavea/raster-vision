from typing import TYPE_CHECKING, Callable, List, Optional
import logging

from rastervision.core.rv_pipeline.rv_pipeline import RVPipeline
from rastervision.core.rv_pipeline.utils import nodata_below_threshold
from rastervision.core.rv_pipeline.object_detection_config import (
    ObjectDetectionWindowMethod)
from rastervision.core.box import Box
from rastervision.core.data.label import ObjectDetectionLabels

if TYPE_CHECKING:
    from rastervision.core.backend.backend import Backend
    from rastervision.core.data import (Labels, Scene, RasterSource,
                                        ObjectDetectionLabelSource)
    from rastervision.core.rv_pipeline.object_detection_config import (
        ObjectDetectionChipOptions)

log = logging.getLogger(__name__)


def _make_chip_pos_windows(image_extent: Box,
                           label_source: 'ObjectDetectionLabelSource',
                           chip_size: int) -> List[Box]:
    chip_size = chip_size
    pos_windows = []
    boxes = label_source.get_labels().get_boxes()
    done_boxes = set()

    # Get a random window around each box. If a box was previously included
    # in a window, then it is skipped.
    for box in boxes:
        if box in done_boxes:
            continue
        # If this  object is bigger than the chip, don't use this box.
        if chip_size < box.width or chip_size < box.height:
            log.warning(f'Label is larger than chip size: {box} '
                        'Skipping this label.')
            continue

        window = box.make_random_square_container(chip_size)
        pos_windows.append(window)

        # Get boxes that lie completely within window
        window_boxes = label_source.get_labels(window=window)
        window_boxes = ObjectDetectionLabels.get_overlapping(
            window_boxes, window, ioa_thresh=1.0)
        window_boxes = window_boxes.get_boxes()
        done_boxes.update(window_boxes)

    return pos_windows


def _make_label_pos_windows(image_extent: Box,
                            label_source: 'ObjectDetectionLabelSource',
                            label_buffer: int) -> List[Box]:
    pos_windows = []
    boxes = label_source.get_labels().get_boxes()
    for box in boxes:
        window = box.buffer(label_buffer, image_extent)
        pos_windows.append(window)

    return pos_windows


def make_pos_windows(
        image_extent: Box, label_source: 'ObjectDetectionLabelSource',
        chip_size: int, window_method: ObjectDetectionWindowMethod,
        label_buffer: Optional[int]) -> List[Box]:
    if window_method == ObjectDetectionWindowMethod.chip:
        return _make_chip_pos_windows(image_extent, label_source, chip_size)
    elif window_method == ObjectDetectionWindowMethod.label:
        if label_buffer is None:
            raise ValueError(
                'label_buffer must be specified if '
                'window_method=ObjectDetectionWindowMethod.label.')
        return _make_label_pos_windows(image_extent, label_source,
                                       label_buffer)
    elif window_method == ObjectDetectionWindowMethod.image:
        return [image_extent.copy()]
    else:
        raise NotImplementedError(f'Window method: {window_method}.')


def make_neg_windows(raster_source: 'RasterSource',
                     label_source: 'ObjectDetectionLabelSource',
                     chip_size: int,
                     nb_windows: int,
                     max_attempts: int,
                     filter_windows: Callable,
                     chip_nodata_threshold: float = 1.) -> List[Box]:
    extent = raster_source.extent
    neg_windows = []
    for _ in range(max_attempts):
        for _ in range(max_attempts):
            window = extent.make_random_square(chip_size)
            if any(filter_windows([window])):
                break
        chip = raster_source.get_chip(window)
        labels = ObjectDetectionLabels.get_overlapping(
            label_source.get_labels(), window, ioa_thresh=0.2)

        # If no labels and not too many nodata pixels, append the chip
        nodata_below_thresh = nodata_below_threshold(
            chip, chip_nodata_threshold, nodata_val=0)
        if len(labels) == 0 and nodata_below_thresh:
            neg_windows.append(window)

        if len(neg_windows) == nb_windows:
            break

    return list(neg_windows)


def get_train_windows(scene: 'Scene',
                      chip_opts: 'ObjectDetectionChipOptions',
                      chip_size: int,
                      chip_nodata_threshold: float = 1.) -> List[Box]:
    raster_source = scene.raster_source
    label_source = scene.label_source

    def filter_windows(windows):
        if scene.aoi_polygons:
            windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
        return windows

    window_method = chip_opts.window_method
    if window_method == ObjectDetectionWindowMethod.sliding:
        stride = chip_size
        windows = raster_source.extent.get_windows(chip_size, stride)
        return list(filter_windows(windows))

    # Make positive windows which contain labels.
    pos_windows = make_pos_windows(raster_source.extent, label_source,
                                   chip_size, chip_opts.window_method,
                                   chip_opts.label_buffer)
    pos_windows = filter_windows(pos_windows)
    nb_pos_windows = len(pos_windows)

    # Make negative windows which do not contain labels.
    # Generate randow windows and save the ones that don't contain
    # any labels. It may take many attempts to generate a single
    # negative window, and could get into an infinite loop in some cases,
    # so we cap the number of attempts.
    if nb_pos_windows:
        nb_neg_windows = round(chip_opts.neg_ratio * nb_pos_windows)
    else:
        nb_neg_windows = 100  # just make some
    max_attempts = 100 * nb_neg_windows
    neg_windows = make_neg_windows(
        raster_source,
        label_source,
        chip_size,
        nb_neg_windows,
        max_attempts,
        filter_windows,
        chip_nodata_threshold=chip_nodata_threshold)

    return pos_windows + neg_windows


class ObjectDetection(RVPipeline):
    def get_train_windows(self, scene: 'Scene') -> List[Box]:
        return get_train_windows(
            scene,
            self.config.chip_options,
            self.config.train_chip_sz,
            chip_nodata_threshold=self.config.chip_nodata_threshold)

    def get_train_labels(self, window: Box,
                         scene: 'Scene') -> ObjectDetectionLabels:
        window_labels = scene.label_source.get_labels(window=window)
        return ObjectDetectionLabels.get_overlapping(
            window_labels,
            window,
            ioa_thresh=self.config.chip_options.ioa_thresh,
            clip=True)

    def predict_scene(self, scene: 'Scene', backend: 'Backend') -> 'Labels':
        # Use strided windowing to ensure that each object is fully visible (ie. not
        # cut off) within some window. This means prediction takes 4x longer for object
        # detection :(
        chip_sz = self.config.predict_chip_sz
        stride = chip_sz // 2
        return backend.predict_scene(scene, chip_sz=chip_sz, stride=stride)

    def post_process_predictions(self, labels: ObjectDetectionLabels,
                                 scene: 'Scene') -> ObjectDetectionLabels:
        return ObjectDetectionLabels.prune_duplicates(
            labels,
            score_thresh=self.config.predict_options.score_thresh,
            merge_thresh=self.config.predict_options.merge_thresh)
