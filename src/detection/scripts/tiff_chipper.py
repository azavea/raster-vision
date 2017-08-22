import json
import argparse
from os import makedirs
from os.path import join, splitext, basename
import csv

import numpy as np
import matplotlib as mpl
mpl.use('Agg') # NOQA
import matplotlib.pyplot as plt
import rasterio
from scipy.misc import imsave
from rtree import index


def get_boxes_from_geojson(json_path, image_dataset):
    with open(json_path, 'r') as json_file:
        geojson = json.load(json_file)

    features = geojson['features']
    boxes = []
    box_to_class_id = {}

    for feature in features:
        polygon = feature['geometry']['coordinates'][0]
        # Convert to pixel coords.
        polygon = np.array([image_dataset.index(p[0], p[1]) for p in polygon])

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
    print('# boxes: {}'.format(len(boxes)))
    np_boxes = np.array(boxes)

    width = np_boxes[:, 2] - np_boxes[:, 0]
    print('width (mean, min, max): ({}, {}, {})'.format(
          np.mean(width), np.min(width), np.max(width)))

    height = np_boxes[:, 3] - np_boxes[:, 1]
    print('height (mean, min, max): ({}, {}, {})'.format(
          np.mean(height), np.min(height), np.max(height)))


def make_debug_plot(output_debug_dir, boxes, box_ind, im):
    # draw rectangle representing box
    debug_im = np.copy(im)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        debug_im[xmin, ymin:ymax, :] = 0
        debug_im[xmax - 1, ymin:ymax, :] = 0
        debug_im[xmin:xmax, ymin, :] = 0
        debug_im[xmin:xmax, ymax - 1, :] = 0

    debug_path = join(
        output_debug_dir, '{}.jpg'.format(box_ind))
    imsave(debug_path, debug_im)


def write_chips_csv(csv_path, chip_rows):
    with open(csv_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
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


def get_random_window(box, chip_size):
    xmin, ymin, xmax, ymax = box
    width = xmax - xmin
    rand_x = int(np.random.uniform(xmin - (chip_size - width), xmin))

    height = ymax - ymin
    rand_y = int(np.random.uniform(ymin - (chip_size - height), ymin))

    return (rand_x, rand_y)


def make_chips(image_path, json_path, output_dir, debug=False,
               chip_size=300):
    '''Make training chips from a GeoTIFF and GeoJSON with detections.'''
    makedirs(output_dir, exist_ok=True)
    output_image_dir = join(output_dir, 'images')
    makedirs(output_image_dir, exist_ok=True)

    output_debug_dir = None
    if debug is not None:
        output_debug_dir = join(output_dir, 'debug')
        makedirs(output_debug_dir, exist_ok=True)

    image_dataset = rasterio.open(image_path)
    boxes, box_to_class_id = get_boxes_from_geojson(json_path, image_dataset)
    # build spatial index of boxes to use for fast intersection test.
    rtree_boxes = make_box_index(boxes)

    print_box_stats(boxes)
    chip_rows = []
    done_boxes = set()

    for chip_ind, anchor_box in enumerate(boxes):
        # if the box is contained in a previous chip, then skip it.
        if anchor_box in done_boxes:
            continue

        # extract random window around anchor_box.
        chip_file_name = '{}.jpg'.format(chip_ind)
        rand_x, rand_y = get_random_window(anchor_box, chip_size)
        window = ((rand_x, rand_x + chip_size), (rand_y, rand_y + chip_size))
        chip_im = np.transpose(
            image_dataset.read(window=window), axes=[1, 2, 0])
        # XXX is this specific to the dataset?
        # bgr-ir
        chip_im = chip_im[:, :, [2, 1, 0]]
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
                row = [chip_file_name, chip_xmin, chip_xmax,
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
                redacted_chip_im[clip_xmin:clip_xmax, clip_ymin:clip_ymax, :] = 0   # noqa

        # save the chip.
        chip_path = join(output_image_dir, chip_file_name)
        imsave(chip_path, redacted_chip_im)
        if debug:
            make_debug_plot(output_debug_dir, chip_boxes, chip_ind,
                            chip_im)

    # save csv.
    chip_csv_path = join(output_dir, 'chips.csv')
    write_chips_csv(chip_csv_path, chip_rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff-path')
    parser.add_argument('--json-path')
    parser.add_argument('--output-dir')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--chip-size', type=int, default=300)
    args = parser.parse_args()

    print('tiff_path: {}'.format(args.tiff_path))
    print('json_path: {}'.format(args.json_path))
    print('output_dir: {}'.format(args.output_dir))
    print('debug: {}'.format(args.debug))
    print('chip_size: {}'.format(args.chip_size))

    return args


if __name__ == '__main__':
    args = parse_args()
    make_chips(args.tiff_path, args.json_path, args.output_dir, args.debug,
               args.chip_size)
