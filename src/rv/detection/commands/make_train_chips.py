from os import makedirs
from os.path import join, dirname
import csv

import click
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import rasterio

from object_detection.utils import label_map_util

from rv.utils import (
    load_window, build_vrt, make_empty_dir,
    get_boxes_from_geojson, save_img, BoxDB, print_box_stats,
    get_random_window_for_box, get_random_window, add_blank_chips)
from rv.detection.commands.settings import (
    planet_channel_order, max_num_classes, temp_root_dir)


def write_chips_csv(csv_path, chip_rows, append_csv=False):
    mode = 'a' if append_csv else 'w'
    with open(csv_path, mode) as csv_file:
        csv_writer = csv.writer(csv_file)
        # Only write header if not appending.
        if not append_csv:
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


def make_pos_chips(image_dataset, chip_size, boxes, classes, chip_dir,
                   chip_label_path, no_partial, redact_partial, channel_order,
                   append_csv):
    box_db = BoxDB(boxes)
    chip_rows = []
    done_boxes = set()
    chip_count = 0

    for chip_ind, anchor_box in enumerate(boxes):
        # if the box is contained in a previous chip, then skip it.
        if tuple(anchor_box) in done_boxes:
            continue

        # extract random window around anchor_box.
        chip_fn = '{}.png'.format(chip_ind)
        # upper left corner of window
        rand_x, rand_y = get_random_window_for_box(
            anchor_box, image_dataset.width, image_dataset.height, chip_size)
        # note: rasterio windows use a different dimension ordering than
        # bounding boxes
        window = ((rand_y, rand_y + chip_size), (rand_x, rand_x + chip_size))
        chip_im = load_window(
            image_dataset, channel_order, window=window)
        redacted_chip_im = np.copy(chip_im)

        # find all boxes inside window (which will be turned into a chip)
        # and transform coordinates so they are in the window frame of
        # reference.
        intersecting_inds = box_db.get_intersecting_box_inds(
            rand_x, rand_y, chip_size)

        # boxes in the chip's frame of reference
        chip_boxes = []
        for intersecting_ind in intersecting_inds:
            intersecting_box = boxes[intersecting_ind]
            ymin, xmin, ymax, xmax = intersecting_box
            chip_box = np.array([ymin - rand_y, xmin - rand_x,
                                 ymax - rand_y, xmax - rand_x])
            chip_ymin, chip_xmin, chip_ymax, chip_xmax = chip_box
            chip_box_class_id = classes[intersecting_ind]

            # box is considered partially contained if less than
            # 3/4 is contained.
            contained_ratio = get_contained_ratio(chip_box, chip_size)
            considered_partial = contained_ratio < 0.75

            chip_ymin, chip_xmin, chip_ymax, chip_xmax = \
                np.clip(chip_box, 0, chip_size).astype(np.int32)
            if not (no_partial and considered_partial):
                row = [chip_fn, chip_ymin, chip_xmin,
                       chip_ymax, chip_xmax, chip_box_class_id]
                chip_rows.append(row)
                chip_boxes.append(chip_box)
                done_boxes.add(tuple(intersecting_box))
            elif redact_partial:
                # else, black out (or redact) the box, since we don't want it
                # to count as a negative example. this could be dangerous if
                # the objects you are trying to detect are black boxes :)
                redacted_chip_im[chip_ymin:chip_ymax, chip_xmin:chip_xmax, :] = 0   # noqa

        # save the chip.
        chip_path = join(chip_dir, chip_fn)
        save_img(chip_path, redacted_chip_im)
        chip_count += 1

    write_chips_csv(chip_label_path, chip_rows, append_csv)
    return chip_count


def make_neg_chips(image_dataset, chip_size, boxes, classes, chip_dir,
                   num_neg_chips, max_attempts, channel_order):
    box_db = BoxDB(boxes)
    neg_chips_count = 0
    attempt_count = 0
    while attempt_count < max_attempts and neg_chips_count < num_neg_chips:
        # extract random window
        rand_x, rand_y = get_random_window(
            image_dataset.width, image_dataset.height, chip_size)

        # check if intersects with any boxes
        intersecting_inds = box_db.get_intersecting_box_inds(
            rand_x, rand_y, chip_size)

        # if no intersection
        if len(intersecting_inds) == 0:
            # extract chip
            # note: row is y, col is x
            window = ((rand_y, rand_y + chip_size),
                      (rand_x, rand_x + chip_size))
            chip_im = load_window(
                image_dataset, channel_order, window=window)

            # if not a blank chip (these are in areas of the VRT with no data)
            if np.any(chip_im != 0):
                # save to disk
                chip_fn = 'neg_{}.png'.format(neg_chips_count)
                chip_path = join(chip_dir, chip_fn)
                save_img(chip_path, chip_im)
                neg_chips_count += 1

        attempt_count += 1

    click.echo('Wrote {} negative chips.'.format(neg_chips_count))
    return neg_chips_count


def make_train_chips_for_image(image_path, json_path, chip_dir,
                               chip_label_path, label_map_path, chip_size,
                               num_neg_chips, max_attempts, no_partial,
                               redact_partial,
                               channel_order, append_csv=False):
    '''Make training chips from a GeoTIFF and GeoJSON with detections.'''
    image_dataset = rasterio.open(image_path)

    label_map = None
    if label_map_path is not None:
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=max_num_classes, use_display_name=True)
        label_map = dict([(category['name'], category['id'])
                          for category in categories])

    boxes, classes, _ = get_boxes_from_geojson(
        json_path, image_dataset, label_map=label_map)
    print_box_stats(boxes)

    num_pos_chips = make_pos_chips(
        image_dataset, chip_size, boxes, classes, chip_dir, chip_label_path,
        no_partial, redact_partial, channel_order, append_csv)

    if num_neg_chips is None:
        num_neg_chips = num_pos_chips
        max_attempts = 100 * num_neg_chips
    neg_count = make_neg_chips(
        image_dataset, chip_size, boxes, classes, chip_dir, num_neg_chips,
        max_attempts, channel_order)

    # We filter out all blank negative chips when generating them in
    # make_neg_chips, since sometimes they dominate and are a waste of time
    # for the model. But we still need some of them, so we add some in at the
    # end.
    blank_neg_ratio = 0.05
    blank_neg_count = max(10, int(blank_neg_ratio * neg_count))
    add_blank_chips(blank_neg_count, chip_size, chip_dir)


def _make_train_chips(image_paths, label_path, chip_dir, chip_label_path,
                      label_map_path, chip_size, num_neg_chips, max_attempts,
                      no_partial, redact_partial, channel_order):
    temp_dir = join(temp_root_dir, 'make_train_chips')
    make_empty_dir(temp_dir)

    vrt_path = join(temp_dir, 'index.vrt')
    build_vrt(vrt_path, image_paths)
    makedirs(chip_dir, exist_ok=True)
    makedirs(dirname(chip_label_path), exist_ok=True)

    click.echo('Making chips...')
    make_train_chips_for_image(
        vrt_path, label_path, chip_dir, chip_label_path,
        label_map_path, chip_size, num_neg_chips, max_attempts,
        no_partial, redact_partial, channel_order)


@click.command()
@click.argument('image_paths', nargs=-1)
@click.argument('label_path')
@click.argument('chip_dir')
@click.argument('chip_label_path')
@click.option('--label-map-path', help='Path to label map')
@click.option('--chip-size', default=300, help='Height and width of each chip')
@click.option('--num-neg-chips', default=None, type=float,
              help='Number of chips without objects to generate per image')
@click.option('--max-attempts', default=None, type=float,
              help='Maximum num of random windows to try per image when ' +
                   'generating negative chips.')
@click.option('--no-partial', is_flag=True, help='Whether to include boxes for ' +
              'partially visible objects')
@click.option('--redact-partial', is_flag=True, help='Whether to black out ' +
              'partially visible objects')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
def make_train_chips(image_paths, label_path, chip_dir, chip_label_path,
                     label_map_path, chip_size, num_neg_chips, max_attempts,
                     no_partial, redact_partial, channel_order):
    """Generate a set of training chips.

    Given imagery and a GeoJSON file with labels in the form of bounding
    boxes, this generates a set of chips centered around the boxes, and a
    CSV file with all the bounding boxes. If no num_neg_chips is provided,
    then there will be one negative chip generated per positive chip.

    Args:
        image_paths: List of TIFF files for training data
        label_path: GeoJSON file with training labels (in lat/lng format)
        chip_dir: Directory of chips
        chip_label_path: CSV file with labels for each chip
    """
    _make_train_chips(image_paths, label_path, chip_dir, chip_label_path,
                      label_map_path, chip_size, num_neg_chips, max_attempts,
                      no_partial, redact_partial, channel_order)


if __name__ == '__main__':
    make_train_chips()
