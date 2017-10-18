import json
from os import makedirs
from os.path import join, dirname
import csv

import click
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import rasterio
from scipy.misc import imsave
from rtree import index

from object_detection.utils import label_map_util

from rv.commands.utils import (
    load_window, build_vrt, download_if_needed, make_temp_dir,
    get_boxes_from_geojson)
from rv.commands.settings import planet_channel_order, max_num_classes


class BoxDB():
    def __init__(self, boxes):
        """Build DB of boxes for fast intersection queries

        Args:
            boxes: [N, 4] numpy array of boxes with cols ymin, xmin, ymax, xmax
        """
        self.boxes = boxes
        self.rtree_idx = index.Index()
        for box_ind, box in enumerate(boxes):
            # rtree order is xmin, ymin, xmax, ymax
            rtree_box = (box[1], box[0], box[3], box[2])
            self.rtree_idx.insert(box_ind, rtree_box)

    def get_intersecting_box_inds(self, x, y, box_size):
        query_box = (x, y, x + box_size, y + box_size)
        intersection_inds = list(self.rtree_idx.intersection(query_box))
        return intersection_inds


def print_box_stats(boxes):
    click.echo('# boxes: {}'.format(len(boxes)))

    ymins, xmins, ymaxs, xmaxs = boxes.T
    width = xmaxs - xmins + 1
    click.echo('width (mean, min, max): ({}, {}, {})'.format(
               np.mean(width), np.min(width), np.max(width)))

    height = ymaxs - ymins + 1
    click.echo('height (mean, min, max): ({}, {}, {})'.format(
               np.mean(height), np.min(height), np.max(height)))


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


def get_random_window_for_box(box, im_width, im_height, chip_size):
    """Get random window in image that contains box.

    Returns: upper-left corner of window
    """
    ymin, xmin, ymax, xmax = box

    # ensure that window doesn't go off the edge of the array.
    width = xmax - xmin
    lb = max(0, xmin - (chip_size - width))
    ub = min(im_width - chip_size, xmin)
    rand_x = int(np.random.uniform(lb, ub))

    height = ymax - ymin
    lb = max(0, ymin - (chip_size - height))
    ub = min(im_height - chip_size, ymin)
    rand_y = int(np.random.uniform(lb, ub))

    return (rand_x, rand_y)


def get_random_window(im_width, im_height, chip_size):
    """Get random window somewhere in image.

    Returns: upper-left corner of window
    """
    rand_x = int(np.random.uniform(0, im_width - chip_size))
    rand_y = int(np.random.uniform(0, im_height - chip_size))
    return (rand_x, rand_y)


def make_pos_chips(image_dataset, chip_size, boxes, classes, chip_dir,
                   chip_label_path, no_partial, channel_order, append_csv):
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

            # if box is wholly contained in the window, then add it to the
            # csv.
            is_contained = (np.all(chip_box >= 0) and
                            np.all(chip_box < chip_size))
            if is_contained or not no_partial:
                row = [chip_fn, chip_ymin, chip_xmin,
                       chip_ymax, chip_xmax, chip_box_class_id]
                chip_rows.append(row)
                chip_boxes.append(chip_box)
                done_boxes.add(tuple(intersecting_box))
            else:
                # else, black out (or redact) the box, since we don't want it
                # to count as a negative example. this could be dangerous if
                # the objects you are trying to detect are black boxes :)
                clip_ymin, clip_xmin, clip_ymax, clip_xmax = \
                    np.clip(chip_box, 0, chip_size).astype(np.int32)
                redacted_chip_im[clip_ymin:clip_ymax, clip_xmin:clip_xmax, :] = 0   # noqa

        # save the chip.
        chip_path = join(chip_dir, chip_fn)
        imsave(chip_path, redacted_chip_im)
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

            # save to disk
            chip_fn = 'neg_{}.png'.format(neg_chips_count)
            chip_path = join(chip_dir, chip_fn)
            imsave(chip_path, chip_im)
            neg_chips_count += 1

        attempt_count += 1

    click.echo('Wrote {} negative chips.'.format(neg_chips_count))


def make_train_chips_for_image(image_path, json_path, chip_dir,
                               chip_label_path, label_map_path, chip_size,
                               num_neg_chips, max_attempts, no_partial,
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
        no_partial, channel_order, append_csv)

    if num_neg_chips is None:
        num_neg_chips = num_pos_chips
        max_attempts = 10 * num_neg_chips
    make_neg_chips(image_dataset, chip_size, boxes, classes, chip_dir,
                   num_neg_chips, max_attempts, channel_order)


def _make_train_chips(image_paths, label_path, chip_dir, chip_label_path,
                      label_map_path, chip_size, num_neg_chips, max_attempts,
                      no_partial, channel_order):
    temp_dir = '/opt/data/temp/make_train_chips'
    make_temp_dir(temp_dir)

    vrt_path = join(temp_dir, 'index.vrt')
    build_vrt(vrt_path, image_paths)
    makedirs(chip_dir, exist_ok=True)
    makedirs(dirname(chip_label_path), exist_ok=True)

    click.echo('Making chips...')
    make_train_chips_for_image(
        vrt_path, label_path, chip_dir, chip_label_path,
        label_map_path, chip_size, num_neg_chips, max_attempts,
        no_partial, channel_order)


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
@click.option('--no-partial', is_flag=True,
              help='Black out objects that are only partially visible in' +
                   ' chips')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
def make_train_chips(image_paths, label_path, chip_dir, chip_label_path,
                     label_map_path, chip_size, num_neg_chips, max_attempts,
                     no_partial, channel_order):
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
                      no_partial, channel_order)


if __name__ == '__main__':
    make_train_chips()
