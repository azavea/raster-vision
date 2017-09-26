import json
from os import makedirs
from os.path import join, splitext, basename, dirname
import csv

import click
import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import rasterio
from scipy.misc import imsave
from rtree import index

from rv.commands.utils import (
    load_window, download_and_build_vrt, download_if_needed, make_temp_dir)
from rv.commands.settings import planet_channel_order


def get_boxes_from_geojson(json_path, image_dataset):
    with open(json_path, 'r') as json_file:
        geojson = json.load(json_file)

    features = geojson['features']
    boxes = []
    box_to_class_id = {}

    for feature in features:
        polygon = feature['geometry']['coordinates'][0]
        # Convert to pixel coords.
        polygon = [image_dataset.index(p[0], p[1]) for p in polygon]
        polygon = np.array([(p[1], p[0]) for p in polygon])

        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        box = (xmin, ymin, xmax, ymax)
        boxes.append(box)

        # Get class_id if exists, else use default of 1.
        class_id = 1
        if 'properties' in feature:
            if 'class_id' in feature['properties']:
                class_id = feature['properties']['class_id']
        box_to_class_id[box] = class_id

    # Remove duplicates. Needed for ships dataset.
    boxes = list(set(boxes))
    return boxes, box_to_class_id


def print_box_stats(boxes):
    click.echo('# boxes: {}'.format(len(boxes)))
    np_boxes = np.array(boxes)

    width = np_boxes[:, 2] - np_boxes[:, 0] + 1
    click.echo('width (mean, min, max): ({}, {}, {})'.format(
               np.mean(width), np.min(width), np.max(width)))

    height = np_boxes[:, 3] - np_boxes[:, 1] + 1
    click.echo('height (mean, min, max): ({}, {}, {})'.format(
               np.mean(height), np.min(height), np.max(height)))


def write_chips_csv(csv_path, chip_rows, append_csv=False):
    mode = 'a' if append_csv else 'w'
    with open(csv_path, mode) as csv_file:
        csv_writer = csv.writer(csv_file)
        # Only write header if not appending.
        if not append_csv:
            csv_writer.writerow(
                ('filename', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'))
        for row in chip_rows:
            csv_writer.writerow(row)


def make_box_index(boxes):
    idx = index.Index()
    for box_ind, box in enumerate(boxes):
        idx.insert(box_ind, box)
    return idx


def find_intersected_boxes(rand_x, rand_y, chip_size, rtree_boxes, boxes):
    box = (rand_x, rand_y, rand_x + chip_size, rand_y + chip_size)
    intersection_ids = list(rtree_boxes.intersection(box))
    return [boxes[id] for id in intersection_ids]


def get_random_window_for_box(box, im_width, im_height, chip_size):
    xmin, ymin, xmax, ymax = box

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
    rand_x = int(np.random.uniform(0, im_width - chip_size))
    rand_y = int(np.random.uniform(0, im_height - chip_size))
    return (rand_x, rand_y)


def make_pos_chips(image_id, image_dataset, chip_size, boxes, rtree_boxes,
                   box_to_class_id, chip_dir, chip_label_path,
                   channel_order, append_csv):
    chip_rows = []
    done_boxes = set()

    for chip_ind, anchor_box in enumerate(boxes):
        # if the box is contained in a previous chip, then skip it.
        if anchor_box in done_boxes:
            continue

        # extract random window around anchor_box.
        chip_fn = '{}_{}.png'.format(image_id, chip_ind)
        rand_x, rand_y = get_random_window_for_box(
            anchor_box, image_dataset.width, image_dataset.height, chip_size)
        window = ((rand_y, rand_y + chip_size), (rand_x, rand_x + chip_size))

        chip_im = load_window(
            image_dataset, channel_order, window=window)
        redacted_chip_im = np.copy(chip_im)

        # find all boxes inside window and transform coordinates so they
        # are in the window frame of reference.
        intersected_boxes = find_intersected_boxes(
            rand_x, rand_y, chip_size, rtree_boxes, boxes)

        chip_boxes = []
        for intersected_box in intersected_boxes:
            xmin, ymin, xmax, ymax = intersected_box
            chip_box = np.array((xmin - rand_x, ymin - rand_y,
                                 xmax - rand_x, ymax - rand_y))
            chip_xmin, chip_ymin, chip_xmax, chip_ymax = chip_box
            chip_class_id = box_to_class_id[intersected_box]

            # if box is wholly contained in the window, then add it to the
            # csv.
            is_contained = (np.all(chip_box >= 0) and
                            np.all(chip_box < chip_size))
            if is_contained:
                row = [chip_fn, chip_xmin, chip_xmax,
                       chip_ymin, chip_ymax, chip_class_id]
                chip_rows.append(row)
                chip_boxes.append(chip_box)
                done_boxes.add(intersected_box)
            else:
                # else, black out the box, since we don't want it to count
                # as a negative example. this could be dangerous if the objects
                # you are trying to detect are black boxes :)
                clip_xmin, clip_ymin, clip_xmax, clip_ymax = \
                    np.clip(chip_box, 0, chip_size)
                redacted_chip_im[clip_ymin:clip_ymax, clip_xmin:clip_xmax, :] = 0   # noqa

        # save the chip.
        chip_path = join(chip_dir, chip_fn)

        imsave(chip_path, redacted_chip_im)

    write_chips_csv(chip_label_path, chip_rows, append_csv)


def make_neg_chips(image_id, image_dataset, chip_size, boxes, rtree_boxes,
                   chip_dir, num_neg_chips, max_attempts, channel_order):
    neg_chips_count = 0
    attempt_count = 0
    while attempt_count < max_attempts and neg_chips_count < num_neg_chips:
        # extract random window
        rand_x, rand_y = get_random_window(
            image_dataset.width, image_dataset.height, chip_size)

        # check if intersects with any boxes
        intersected_boxes = find_intersected_boxes(
            rand_x, rand_y, chip_size, rtree_boxes, boxes)

        # if no intersection
        if len(intersected_boxes) == 0:
            # extract chip
            window = ((rand_y, rand_y + chip_size),
                      (rand_x, rand_x + chip_size))
            chip_im = load_window(
                image_dataset, channel_order, window=window)

            # save to disk
            chip_fn = '{}_neg_{}.png'.format(image_id, neg_chips_count)
            chip_path = join(chip_dir, chip_fn)
            imsave(chip_path, chip_im)

            neg_chips_count += 1
        attempt_count += 1
    click.echo('Wrote {} negative chips.'.format(neg_chips_count))


def make_train_chips_for_image(image_path, image_id, json_path, chip_dir,
                               chip_label_path, chip_size, num_neg_chips,
                               max_attempts, channel_order, append_csv=False):
    '''Make training chips from a GeoTIFF and GeoJSON with detections.'''
    image_dataset = rasterio.open(image_path)
    boxes, box_to_class_id = get_boxes_from_geojson(json_path, image_dataset)
    print_box_stats(boxes)
    # build spatial index of boxes to use for fast intersection test.
    rtree_boxes = make_box_index(boxes)

    make_pos_chips(
        image_id, image_dataset, chip_size, boxes, rtree_boxes,
        box_to_class_id, chip_dir, chip_label_path, channel_order,
        append_csv)

    make_neg_chips(image_id, image_dataset, chip_size, boxes, rtree_boxes,
                   chip_dir, num_neg_chips, max_attempts, channel_order)


@click.command()
@click.argument('image_uris', nargs=-1)
@click.argument('label_uri')
@click.argument('chip_dir')
@click.argument('chip_label_path')
@click.option('--chip-size', default=300, help='Height and width of each chip')
@click.option('--num-neg-chips', default=0,
              help='Number of chips without objects to generate per image')
@click.option('--max-attempts', default=0,
              help='Maximum num of random windows to try per image when ' +
                   'generating negative chips.')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
def make_train_chips(image_uris, label_uri, chip_dir, chip_label_path,
                     chip_size, num_neg_chips, max_attempts, channel_order):
    """Generate a set of training chips.

    Given imagery and a GeoJSON file with labels in the form of bounding
    boxes, this generates a set of chips centered around the boxes, and a
    CSV file with all the bounding boxes.

    Args:
        image_uris: List of URIs of TIFF files for training data
        label_uri: GeoJSON file with training labels
        chip_dir: Directory of chips
        chip_label_path: CSV file with labels for each chip
    """
    temp_dir = '/opt/data/temp/'
    make_temp_dir(temp_dir)

    image_path = download_and_build_vrt(temp_dir, image_uris)
    image_fn = basename(image_path)
    image_id = splitext(image_fn)[0]
    label_path = download_if_needed(temp_dir, label_uri)

    makedirs(chip_dir, exist_ok=True)
    makedirs(dirname(chip_label_path), exist_ok=True)

    click.echo('Making chips for {}...'.format(image_fn))
    make_train_chips_for_image(
        image_path, image_id, label_path, chip_dir, chip_label_path,
        chip_size, num_neg_chips, max_attempts, channel_order)


if __name__ == '__main__':
    make_train_chips()
