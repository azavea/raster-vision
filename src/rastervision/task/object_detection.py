import numpy as np
import logging

from rastervision.task import Task
from rastervision.data import ObjectDetectionLabels
from rastervision.core import Box

log = logging.getLogger(__name__)


def _make_chip_pos_windows(image_extent, label_store, chip_size):
    chip_size = chip_size
    pos_windows = []
    boxes = label_store.get_labels().get_boxes()
    done_boxes = set()

    # Get a random window around each box. If a box was previously included
    # in a window, then it is skipped.
    for box in boxes:
        if box.tuple_format() not in done_boxes:
            # If this  object is bigger than the chip,
            # don't use this box.
            if chip_size < box.get_width() or chip_size < box.get_height():
                log.warning('Label is larger than chip size: {} '
                            'Skipping this label'.format(box.tuple_format()))
                continue

            window = box.make_random_square_container(chip_size)
            pos_windows.append(window)

            # Get boxes that lie completely within window
            window_boxes = label_store.get_labels(window=window)
            window_boxes = ObjectDetectionLabels.get_overlapping(
                window_boxes, window, ioa_thresh=1.0)
            window_boxes = window_boxes.get_boxes()
            window_boxes = [box.tuple_format() for box in window_boxes]
            done_boxes.update(window_boxes)

    return pos_windows


def _make_label_pos_windows(image_extent, label_store, label_buffer):
    pos_windows = []
    for box in label_store.get_labels().get_boxes():
        window = box.make_buffer(label_buffer, image_extent)
        pos_windows.append(window)

    return pos_windows


def make_pos_windows(image_extent, label_store, chip_size, window_method,
                     label_buffer):
    if window_method == 'label':
        return _make_label_pos_windows(image_extent, label_store, label_buffer)
    elif window_method == 'image':
        return [image_extent.make_copy()]
    else:
        return _make_chip_pos_windows(image_extent, label_store, chip_size)


def make_neg_windows(raster_source, label_store, chip_size, nb_windows,
                     max_attempts, filter_windows):
    extent = raster_source.get_extent()
    neg_windows = []
    for _ in range(max_attempts):
        for _ in range(max_attempts):
            window = extent.make_random_square(chip_size)
            if any(filter_windows([window])):
                break
        chip = raster_source.get_chip(window)
        labels = ObjectDetectionLabels.get_overlapping(
            label_store.get_labels(), window, ioa_thresh=0.2)

        # If no labels and not blank, append the chip
        if len(labels) == 0 and np.sum(chip.ravel()) > 0:
            neg_windows.append(window)

        if len(neg_windows) == nb_windows:
            break

    return list(neg_windows)


class ObjectDetection(Task):
    def get_train_windows(self, scene):
        raster_source = scene.raster_source
        label_store = scene.ground_truth_label_source

        def filter_windows(windows):
            if scene.aoi_polygons:
                windows = Box.filter_by_aoi(windows, scene.aoi_polygons)
            return windows

        window_method = self.config.chip_options.window_method
        if window_method == 'sliding':
            chip_size = self.config.chip_size
            stride = chip_size
            return list(
                filter_windows((raster_source.get_extent().get_windows(
                    chip_size, stride))))

        # Make positive windows which contain labels.
        pos_windows = filter_windows(
            make_pos_windows(raster_source.get_extent(), label_store,
                             self.config.chip_size,
                             self.config.chip_options.window_method,
                             self.config.chip_options.label_buffer))
        nb_pos_windows = len(pos_windows)

        # Make negative windows which do not contain labels.
        # Generate randow windows and save the ones that don't contain
        # any labels. It may take many attempts to generate a single
        # negative window, and could get into an infinite loop in some cases,
        # so we cap the number of attempts.
        if nb_pos_windows:
            nb_neg_windows = round(
                self.config.chip_options.neg_ratio * nb_pos_windows)
        else:
            nb_neg_windows = 100  # just make some
        max_attempts = 100 * nb_neg_windows
        neg_windows = make_neg_windows(raster_source, label_store,
                                       self.config.chip_size, nb_neg_windows,
                                       max_attempts, filter_windows)

        return pos_windows + neg_windows

    def get_train_labels(self, window, scene):
        window_labels = scene.ground_truth_label_source.get_labels(
            window=window)
        return ObjectDetectionLabels.get_overlapping(
            window_labels,
            window,
            ioa_thresh=self.config.chip_options.ioa_thresh,
            clip=True)

    def get_predict_windows(self, extent):
        chip_size = self.config.chip_size
        stride = chip_size // 2
        return extent.get_windows(chip_size, stride)

    def post_process_predictions(self, labels, scene):
        return ObjectDetectionLabels.prune_duplicates(
            labels,
            score_thresh=self.config.predict_options.score_thresh,
            merge_thresh=self.config.predict_options.merge_thresh)

    def save_debug_predict_image(self, scene, debug_dir_uri):
        # TODO implement this
        pass
