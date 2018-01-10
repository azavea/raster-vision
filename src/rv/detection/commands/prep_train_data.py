from os.path import splitext, join
import shutil
import tempfile

import click

from rv.detection.commands.make_label_map import _make_label_map
from rv.detection.commands.make_train_chips import _make_train_chips
from rv.detection.commands.make_tf_record import _make_tf_record
from rv.detection.commands.transform_geojson import _transform_geojson
from rv.detection.commands.settings import (
    default_channel_order, temp_root_dir)
from rv.utils.misc import load_projects
from rv.utils.files import (
    download_if_needed, make_dir, get_local_path, upload_if_needed,
    MyTemporaryDirectory)


def filter_annotations(annotations_paths, temp_dir, min_area, single_label):
    filtered_annotations_dir = join(temp_dir, 'filtered_annotations')
    make_dir(filtered_annotations_dir)
    filtered_annotations_paths = []
    for annotation_ind, annotations_path in enumerate(annotations_paths):
        filtered_annotations_path = join(
            filtered_annotations_dir, '{}.json'.format(annotation_ind))
        _transform_geojson(
            annotations_path, filtered_annotations_path, min_area=min_area,
            single_label=single_label)
        filtered_annotations_paths.append(filtered_annotations_path)
    return filtered_annotations_paths


@click.command()
@click.argument('projects_uri')
@click.argument('output_zip_uri')
@click.argument('label_map_uri')
@click.option('--chip-size', default=300, help='Height and width of each chip')
@click.option('--num-neg-chips', default=None, type=float,
              help='Number of chips without objects to generate per image')
@click.option('--max-attempts', default=None, type=float,
              help='Maximum num of random windows to try per image when ' +
                   'generating negative chips.')
@click.option('--channel-order', nargs=3, type=int, default=default_channel_order,
              help='Indices of the RGB channels')
@click.option('--debug', is_flag=True,
              help='Generate debug plots that contain bounding boxes')
@click.option('--min-area', default=0.0,
              help='Minimum allowed area of objects in meters^2')
@click.option('--single-label', help='The label to convert all labels to')
@click.option('--no-partial', is_flag=True, help='Whether to include boxes for ' +
              'partially visible objects')
@click.option('--redact-partial', is_flag=True, help='Whether to black out ' +
              'partially visible objects')
@click.option('--save-temp', is_flag=True)
def prep_train_data(projects_uri, output_zip_uri, label_map_uri, chip_size,
                    num_neg_chips, max_attempts, channel_order, debug,
                    min_area, single_label, no_partial, redact_partial,
                    save_temp):
    """Generate training chips and TFRecord for set of projects.

    Given a set of projects (each a set of images and a GeoJSON file with
    labels), this generates training chips, a TFRecord, and zips them.

    Args:
        projects_uri: JSON file listing projects
            (each a list of images and an annotation file)
        output_zip_uri: zip file that will contain the training data
    """
    prefix = temp_root_dir
    temp_dir = join(prefix, 'prep-train-data') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        if num_neg_chips is not None and max_attempts is None:
            max_attempts = 10 * num_neg_chips

        projects_path = download_if_needed(projects_uri, temp_dir)
        project_ids, image_paths_list, annotations_paths = \
            load_projects(projects_path, temp_dir)
        annotations_paths = filter_annotations(
            annotations_paths, temp_dir, min_area, single_label)

        output_zip_path = get_local_path(output_zip_uri, temp_dir)
        output_zip_dir = splitext(output_zip_path)[0]
        make_dir(output_zip_dir, check_empty=True)

        label_map_path = get_local_path(label_map_uri, temp_dir)
        make_dir(label_map_path, use_dirname=True)
        _make_label_map(annotations_paths, label_map_path)

        train_chip_dir = join(temp_dir, 'train-chips')
        chip_dirs = []
        chip_label_paths = []
        for project_id, image_paths, annotations_path in \
                zip(project_ids, image_paths_list, annotations_paths):
            chip_dir = join(train_chip_dir, str(project_id))
            chip_dirs.append(chip_dir)
            chip_label_path = join(train_chip_dir, '{}.csv'.format(project_id))
            chip_label_paths.append(chip_label_path)

            _make_train_chips(image_paths, annotations_path, chip_dir,
                              chip_label_path, label_map_path, chip_size,
                              num_neg_chips, max_attempts, no_partial,
                              redact_partial, channel_order, save_temp)

        _make_tf_record(label_map_path, chip_dirs, chip_label_paths,
                        output_zip_dir, debug)

        # Copy label map so it's included in the zip file for convenience.
        label_map_copy_path = join(output_zip_dir, 'label-map.pbtxt')
        shutil.copyfile(label_map_path, label_map_copy_path)
        shutil.make_archive(output_zip_dir, 'zip', output_zip_dir)

        upload_if_needed(output_zip_path, output_zip_uri)
        upload_if_needed(label_map_path, label_map_uri)


if __name__ == '__main__':
    prep_train_data()
