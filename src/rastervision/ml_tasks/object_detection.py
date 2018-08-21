import numpy as np
import random

from object_detection.utils import visualization_utils as vis_util

from rastervision.core.ml_task import MLTask
from rastervision.core.box import Box
from rastervision.evaluations.object_detection_evaluation import (
    ObjectDetectionEvaluation)
from rastervision.labels.object_detection_labels import ObjectDetectionLabels
from rastervision.utils.misc import save_img


def save_debug_image(im, labels, class_map, output_path):
    npboxes = labels.get_npboxes()
    class_ids = labels.get_class_ids()
    scores = labels.get_scores()
    if scores is None:
        scores = [1.0] * len(labels)

    vis_util.visualize_boxes_and_labels_on_image_array(
        im,
        npboxes,
        class_ids,
        scores,
        class_map.get_category_index(),
        use_normalized_coordinates=True,
        line_thickness=2,
        max_boxes_to_draw=None)
    save_img(im, output_path)


def _make_chip_pos_windows(image_extent, label_store, options):
    chip_size = options.chip_size
    pos_windows = []
    boxes = label_store.get_labels().get_boxes()
    done_boxes = set()

    # Get a random window around each box. If a box was previously included
    # in a window, then it is skipped.
    for box in boxes:
        if box.tuple_format() not in done_boxes:
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


def _make_label_pos_windows(image_extent, label_store, options):
    label_buffer = options.object_detection_options.label_buffer
    pos_windows = []
    for box in label_store.get_labels().get_boxes():
        window = box.make_buffer(label_buffer, image_extent)
        pos_windows.append(window)

    return pos_windows


def make_pos_windows(image_extent, label_store, options):
    window_method = options.object_detection_options.window_method

    if window_method == 'label':
        return _make_label_pos_windows(image_extent, label_store, options)
    elif window_method == 'image':
        return [image_extent.make_copy()]
    else:
        return _make_chip_pos_windows(image_extent, label_store, options)


def make_neg_windows(raster_source, label_store, chip_size, nb_windows,
                     max_attempts):
    extent = raster_source.get_extent()
    neg_windows = []
    for _ in range(max_attempts):
        window = extent.make_random_square(chip_size)
        chip = raster_source.get_chip(window)
        labels = ObjectDetectionLabels.get_overlapping(
            label_store.get_labels(), window, ioa_thresh=0.2)

        # If no labels and not blank, append the chip
        if len(labels) == 0 and np.sum(chip.ravel()) > 0:
            neg_windows.append(window)

        if len(neg_windows) == nb_windows:
            break

    return neg_windows


class ObjectDetection(MLTask):
    def transform_training_chip(self, window, chip, labels, options):
        nodata_aug_prob = options.object_detection_options.nodata_aug_prob

        # If negative chip, with some probability, add a random black square
        # to chip.
        if len(labels) == 0 and random.uniform(0, 1) < nodata_aug_prob:
            size = round(random.uniform(0, 1) * chip.shape[0])
            square = Box(0, 0, chip.shape[0],
                         chip.shape[1]).make_random_square(size)
            chip = np.copy(chip)
            chip[square.ymin:square.ymax, square.xmin:square.xmax, :] = 0
        return chip, labels

    def get_train_windows(self, scene, options):
        raster_source = scene.raster_source
        label_store = scene.ground_truth_label_store

        window_method = options.object_detection_options.window_method
        if window_method == 'sliding':
            chip_size = options.chip_size
            stride = chip_size
            return list(raster_source.get_extent().get_windows(
                chip_size, stride))

        # Make positive windows which contain labels.
        pos_windows = make_pos_windows(raster_source.get_extent(), label_store,
                                       options)
        nb_pos_windows = len(pos_windows)

        # Make negative windows which do not contain labels.
        # Generate randow windows and save the ones that don't contain
        # any labels. It may take many attempts to generate a single
        # negative window, and could get into an infinite loop in some cases,
        # so we cap the number of attempts.
        if nb_pos_windows:
            nb_neg_windows = round(
                options.object_detection_options.neg_ratio * nb_pos_windows)
        else:
            nb_neg_windows = 100  # just make some
        max_attempts = 100 * nb_neg_windows
        neg_windows = make_neg_windows(raster_source, label_store,
                                       options.chip_size, nb_neg_windows,
                                       max_attempts)

        return pos_windows + neg_windows

    def get_train_labels(self, window, scene, options):
        window_labels = scene.ground_truth_label_store.get_labels(
            window=window)
        return ObjectDetectionLabels.get_overlapping(
            window_labels,
            window,
            ioa_thresh=options.object_detection_options.ioa_thresh,
            clip=True)

    def get_predict_windows(self, extent, options):
        chip_size = options.chip_size
        stride = chip_size // 2
        return extent.get_windows(chip_size, stride)

    def post_process_predictions(self, labels, options):
        return ObjectDetectionLabels.prune_duplicates(
            labels,
            score_thresh=options.object_detection_options.score_thresh,
            merge_thresh=options.object_detection_options.merge_thresh)

    def get_evaluation(self):
        return ObjectDetectionEvaluation()

    def save_debug_predict_image(self, scene, debug_dir_uri):
        # TODO implement this
        pass
