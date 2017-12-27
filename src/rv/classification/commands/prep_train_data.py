from os.path import splitext, join
import shutil
import glob

import rasterio
import click
import numpy as np

from rv.utils import (
    download_if_needed, make_empty_dir, get_local_path, upload_if_needed,
    load_projects, BoxDB, get_random_window_for_box, get_random_window,
    load_window, save_img, get_boxes_from_geojson,
    print_box_stats, build_vrt, add_blank_chips)
from rv.classification.commands.settings import (
    planet_channel_order, temp_root_dir)


def make_pos_chips(project_ind, image_dataset, chip_size, boxes, classes,
                   pos_dir, nb_pos_sets, channel_order):
    # For each box, extract a randomly located window that contains it.
    for pos_set_ind in range(nb_pos_sets):
        for chip_ind, box in enumerate(boxes):
            # upper left corner of window
            rand_x, rand_y = get_random_window_for_box(
                box, image_dataset.width, image_dataset.height, chip_size)

            # note: rasterio windows use a different dimension ordering than
            # bounding boxes
            window = ((rand_y, rand_y + chip_size),
                      (rand_x, rand_x + chip_size))
            chip_im = load_window(
                image_dataset, channel_order, window=window)

            chip_fn = '{}-{}-{}.png'.format(project_ind, pos_set_ind, chip_ind)
            chip_path = join(pos_dir, chip_fn)
            save_img(chip_path, chip_im)

    pos_count = len(boxes) * nb_pos_sets
    return pos_count


def make_neg_chips(project_ind, desired_neg_count,
                   image_dataset, chip_size, boxes, classes, neg_dir,
                   channel_order):
    box_db = BoxDB(boxes)
    neg_count = 0
    max_attempts = desired_neg_count * 100

    for _ in range(max_attempts):
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
                chip_fn = '{}-{}.png'.format(project_ind, neg_count)
                chip_path = join(neg_dir, chip_fn)
                save_img(chip_path, chip_im)
                neg_count += 1

        if neg_count == desired_neg_count:
            break

    return neg_count


def process_image(project_ind, image_path, annotations_path, pos_dir, neg_dir,
                  chip_size, nb_pos_sets, neg_ratio, channel_order):
    '''Make training chips from a GeoTIFF and GeoJSON with detections.'''
    image_dataset = rasterio.open(image_path)

    boxes, classes, _ = get_boxes_from_geojson(annotations_path, image_dataset)
    print_box_stats(boxes)

    pos_count = make_pos_chips(
        project_ind, image_dataset, chip_size, boxes, classes, pos_dir,
        nb_pos_sets, channel_order)
    print('Wrote {} positive chips.'.format(pos_count))

    desired_neg_count = int(neg_ratio * pos_count)
    neg_count = make_neg_chips(
        project_ind, desired_neg_count, image_dataset, chip_size, boxes,
        classes, neg_dir, channel_order)
    print('Wrote {} negative chips.'.format(neg_count))
    print()


@click.command()
@click.argument('projects_uri')
@click.argument('output_zip_uri')
@click.option('--chip-size', default=128, help='Height and width of each chip')
@click.option('--nb-pos-sets', default=2,
              help='Number of positive chips to generate per object')
@click.option('--neg-ratio', default=1,
              help='Ratio of negative to positive chips')
@click.option('--blank-neg-ratio', default=0.05,
              help='Ratio of blank to non-blank negative chips')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
def prep_train_data(projects_uri, output_zip_uri, chip_size,
                    nb_pos_sets, neg_ratio, blank_neg_ratio, channel_order):
    """Generate training chips and label map for set of projects.

    Given a set of projects (each a set of images and a GeoJSON file with
    labels), this generates training chips and zips them.

    Args:
        projects_uri: JSON file listing projects
            (each a list of images and an annotation file)
        output_zip_uri: zip file that will contain the training data
    """
    temp_dir = join(temp_root_dir, 'prep_train_data')
    make_empty_dir(temp_dir)

    projects_path = download_if_needed(temp_dir, projects_uri)
    image_paths_list, annotations_paths = \
        load_projects(temp_dir, projects_path)

    output_zip_path = get_local_path(temp_dir, output_zip_uri)
    output_zip_dir = splitext(output_zip_path)[0]
    make_empty_dir(output_zip_dir)

    pos_dir = join(output_zip_dir, 'pos')
    make_empty_dir(pos_dir)
    neg_dir = join(output_zip_dir, 'neg')
    make_empty_dir(neg_dir)

    for project_ind, (image_paths, annotations_path) in \
            enumerate(zip(image_paths_list, annotations_paths)):
        print('Processing project {}'.format(project_ind))
        vrt_path = join(temp_dir, 'index.vrt')
        build_vrt(vrt_path, image_paths)

        process_image(
            project_ind, vrt_path, annotations_path, pos_dir, neg_dir,
            chip_size, nb_pos_sets, neg_ratio, channel_order)

    # We filter out all blank negative chips when generating them in
    # make_neg_chips, since sometimes they dominate and are a waste of time
    # for the model. But we still need some of them, so we add some in at the
    # end.
    neg_count = len(glob.glob(join(neg_dir, '*.png')))
    blank_neg_count = max(10, int(blank_neg_ratio * neg_count))
    add_blank_chips(blank_neg_count, chip_size, neg_dir)

    # Copy label map so it's included in the zip file for convenience.
    # label_map_copy_path = join(output_zip_dir, 'label-map.pbtxt')
    # shutil.copyfile(label_map_path, label_map_copy_path)
    # upload_if_needed(label_map_path, label_map_uri)
    shutil.make_archive(output_zip_dir, 'zip', output_zip_dir)
    upload_if_needed(output_zip_path, output_zip_uri)


if __name__ == '__main__':
    prep_train_data()
