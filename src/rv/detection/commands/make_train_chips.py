from os.path import join
import csv

import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import rasterio

from object_detection.utils import label_map_util

from rv.utils.geo import (
    load_window, build_vrt, get_boxes_from_geojson, BoxDB, print_box_stats,
    get_random_window_for_box, get_random_window)
from rv.utils.files import make_dir, MyTemporaryDirectory
from rv.utils.misc import save_img
from rv.detection.commands.settings import (
    max_num_classes, temp_root_dir)


def write_chips_csv(csv_path, chip_rows):
    # Write header and rows to CSV file.
    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ('filename', 'ymin', 'xmin', 'ymax', 'xmax', 'class_id'))
        for row in chip_rows:
            csv_writer.writerow(row)


def get_box_area(box):
    ymin, xmin, ymax, xmax = box
    return (ymax - ymin) * (xmax - xmin)


def get_contained_ratio(chip_box, chip_size):
    clipped_chip_box = np.clip(chip_box, 0, chip_size)
    contained_area = get_box_area(clipped_chip_box)
    area = get_box_area(chip_box)
    return contained_area / area


def make_pos_chips(image_dataset, boxes, classes, chip_dir, chip_label_path,
                   config):
    box_db = BoxDB(boxes)
    chip_rows = []
    chip_count = 0
    chip_size = config.chip_size

    for chip_ind, anchor_box in enumerate(boxes):
        # extract random window around anchor_box.
        chip_fn = '{}.png'.format(chip_ind)
        # upper left corner of window
        rand_x, rand_y = get_random_window_for_box(
            anchor_box, image_dataset.width, image_dataset.height, chip_size)
        # note: rasterio windows use a different dimension ordering than
        # bounding boxes
        window = ((rand_y, rand_y + chip_size), (rand_x, rand_x + chip_size))
        chip_im = load_window(
            image_dataset, config.channel_order, window=window)

        # find all boxes inside window and transform coordinates so they are
        # in the window frame of reference.
        intersecting_inds = box_db.get_intersecting_box_inds(
            rand_x, rand_y, chip_size)

        for intersecting_ind in intersecting_inds:
            intersecting_box = boxes[intersecting_ind]
            ymin, xmin, ymax, xmax = intersecting_box
            chip_box = np.array([ymin - rand_y, xmin - rand_x,
                                 ymax - rand_y, xmax - rand_x])
            chip_ymin, chip_xmin, chip_ymax, chip_xmax = chip_box
            chip_box_class_id = classes[intersecting_ind]

            # box is considered fully contained if > 0.75 is contained in the
            # window.
            contained_ratio = get_contained_ratio(chip_box, chip_size)
            if contained_ratio > 0.75:
                # clip the box so it lies within window.
                chip_ymin, chip_xmin, chip_ymax, chip_xmax = \
                    np.clip(chip_box, 0, chip_size).astype(np.int32)

                row = [chip_fn, chip_ymin, chip_xmin,
                       chip_ymax, chip_xmax, chip_box_class_id]
                chip_rows.append(row)

        # save the chip.
        chip_path = join(chip_dir, chip_fn)
        save_img(chip_path, chip_im)
        chip_count += 1

    # Write all boxes to CSV file.
    write_chips_csv(chip_label_path, chip_rows)
    return chip_count


def make_neg_chips(image_dataset, boxes, classes, chip_dir, desired_neg_chips,
                   config):
    box_db = BoxDB(boxes)
    max_attempts = 100 * desired_neg_chips
    neg_chips_count = 0
    attempt_count = 0
    chip_size = config.chip_size

    # Try to collect num_neg_chips negative chips.
    while attempt_count < max_attempts and neg_chips_count < desired_neg_chips:
        # Extract random window.
        rand_x, rand_y = get_random_window(
            image_dataset.width, image_dataset.height, chip_size)

        # Check if intersects with any boxes.
        intersecting_inds = box_db.get_intersecting_box_inds(
            rand_x, rand_y, chip_size)

        # If no intersection, then extract chip.
        if len(intersecting_inds) == 0:
            # note: row is y, col is x
            window = ((rand_y, rand_y + chip_size),
                      (rand_x, rand_x + chip_size))
            chip_im = load_window(
                image_dataset, config.channel_order, window=window)

            # If more than half of chip has data (ie. not zero),
            # then save to disk.
            if np.mean(np.ravel(chip_im != 0)) > 0.5:
                chip_fn = 'neg_{}.png'.format(neg_chips_count)
                chip_path = join(chip_dir, chip_fn)
                save_img(chip_path, chip_im)
                neg_chips_count += 1

        attempt_count += 1

    return neg_chips_count


def _make_train_chips(image_path, annotations_path, chip_dir, chip_label_path,
                      label_map_path, config):
    '''Make training chips from a GeoTIFF and GeoJSON with detections.'''
    print('Making chips...')
    image_dataset = rasterio.open(image_path)

    # Load label map
    label_map = None
    if label_map_path is not None:
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=max_num_classes, use_display_name=True)
        label_map = dict([(category['name'], category['id'])
                          for category in categories])

    # Load boxes and corresponding classes from GeoJSON file.
    boxes, classes, _ = get_boxes_from_geojson(
        annotations_path, image_dataset, label_map=label_map)
    print_box_stats(boxes)

    # TODO validate that boxes are inside bounds of image_dataset

    # Make positive chips (ie. those with boxes in them).
    num_pos_chips = make_pos_chips(
        image_dataset, boxes, classes, chip_dir, chip_label_path,
        config)

    # Make negative chips (ie. those without any boxes in them).
    desired_neg_chips = int(num_pos_chips * config.neg_ratio)
    num_neg_chips = make_neg_chips(
        image_dataset, boxes, classes, chip_dir, desired_neg_chips, config)

    print('Wrote {} pos and {} neg chips.'.format(
        num_pos_chips, num_neg_chips))


def make_train_chips(image_paths, annotations_path, chip_dir,
                     chip_label_path, label_map_path, config,
                     save_temp):
    # Setup directories.
    prefix = temp_root_dir
    temp_dir = join(prefix, 'make-train-chips') if save_temp else None
    make_dir(chip_dir, check_empty=True)
    make_dir(chip_label_path, use_dirname=True)

    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        # Create VRT from list of images and make chips.
        vrt_path = join(temp_dir, 'index.vrt')
        build_vrt(vrt_path, image_paths)

        _make_train_chips(
            vrt_path, annotations_path, chip_dir, chip_label_path,
            label_map_path, config)
