from os.path import splitext, join
from os import makedirs
import shutil
import json

import click

from rv.commands.make_label_map import _make_label_map
from rv.commands.make_train_chips import _make_train_chips
from rv.commands.make_tf_record import _make_tf_record
from rv.commands.utils import (
    download_if_needed, make_temp_dir, get_local_path, upload_if_needed)
from rv.commands.settings import planet_channel_order


def load_projects(temp_dir, projects_path):
    image_paths_list = []
    annotations_paths = []
    with open(projects_path, 'r') as projects_file:
        projects = json.load(projects_file)
        for project in projects:
            image_uris = project['images']
            image_paths = [download_if_needed(temp_dir, image_uri)
                           for image_uri in image_uris]
            image_paths_list.append(image_paths)
            annotations_uri = project['annotations']
            annotations_path = download_if_needed(temp_dir, annotations_uri)
            annotations_paths.append(annotations_path)

    return image_paths_list, annotations_paths


@click.command()
@click.argument('projects_uri')
@click.argument('output_zip_uri')
@click.option('--chip-size', default=300, help='Height and width of each chip')
@click.option('--num-neg-chips', default=0,
              help='Number of chips without objects to generate per image')
@click.option('--max-attempts', default=0,
              help='Maximum num of random windows to try per image when ' +
                   'generating negative chips.')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
@click.option('--debug', is_flag=True,
              help='Generate debug plots that contain bounding boxes')
def prep_train_data(projects_uri, output_zip_uri, chip_size,
                    num_neg_chips, max_attempts, channel_order, debug):
    """Generate training chips and TFRecord for set of projects.

    Given a set of projects (each a set of images and a GeoJSON file with
    labels), this generates training chips, a TFRecord, and zips them.

    Args:
        projects_uri: JSON file listing projects
            (each a list of images and an annotation file)
        output_zip_uri: zip file that will contain the training data
    """
    temp_dir = '/opt/data/temp/'
    make_temp_dir(temp_dir)

    projects_path = download_if_needed(temp_dir, projects_uri)
    image_paths_list, annotations_paths = \
        load_projects(temp_dir, projects_path)

    output_zip_path = get_local_path(temp_dir, output_zip_uri)
    output_zip_dir = splitext(output_zip_path)[0]
    makedirs(output_zip_dir, exist_ok=True)

    label_map_path = join(output_zip_dir, 'label_map.pbtxt')

    _make_label_map(annotations_paths, label_map_path)

    train_chip_dir = join(temp_dir, 'train_chips')
    chip_dirs = []
    chip_label_paths = []
    for project_ind, (image_paths, annotations_path) in \
            enumerate(zip(image_paths_list, annotations_paths)):
        chip_dir = join(train_chip_dir, str(project_ind))
        chip_dirs.append(chip_dir)
        chip_label_path = join(train_chip_dir, '{}.csv'.format(project_ind))
        chip_label_paths.append(chip_label_path)

        _make_train_chips(image_paths, annotations_path, chip_dir,
                          chip_label_path, label_map_path, chip_size,
                          num_neg_chips, max_attempts, channel_order)

    _make_tf_record(label_map_path, chip_dirs, chip_label_paths,
                    output_zip_dir, debug)

    shutil.make_archive(output_zip_dir, 'zip')
    upload_if_needed(output_zip_path, output_zip_uri)


if __name__ == '__main__':
    prep_train_data()
