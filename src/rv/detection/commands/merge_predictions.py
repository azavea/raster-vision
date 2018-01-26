import os
import json
import copy

import click

from rv.detection.commands.settings import (
    temp_root_dir)
from rv.utils.files import (
    download_if_needed, get_local_path, upload_if_needed,
    MyTemporaryDirectory)


def get_annotations_paths(projects_path, temp_dir):
    annotations_paths = []
    with open(projects_path, 'r') as projects_file:
        projects = json.load(projects_file)
        for project_ind, project in enumerate(projects):
            annotations_uri = project['annotations']
            annotations_path = download_if_needed(
                annotations_uri, temp_dir)
            annotations_paths.append(annotations_path)
    return annotations_paths


def merge_annotations(annotations_list):
    all_annotations = copy.deepcopy(annotations_list[0])
    for annotations in annotations_list[1:]:
        all_annotations['features'].extend(annotations['features'])
    return all_annotations


@click.command()
@click.argument('projects_uri')
@click.argument('output_uri')
@click.option('--save-temp', is_flag=True)
def merge_predictions(projects_uri, output_uri, save_temp):
    prefix = temp_root_dir
    temp_dir = os.path.join(prefix, 'merge-predictions') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        projects_path = download_if_needed(projects_uri, temp_dir)
        output_path = get_local_path(output_uri, temp_dir)

        annotation_paths = get_annotations_paths(projects_path, temp_dir)
        annotations_list = []
        for annotation_path in annotation_paths:
            with open(annotation_path, 'r') as annotation_file:
                annotations_list.append(json.load(annotation_file))

        annotations = merge_annotations(annotations_list)
        with open(output_path, 'w') as output_file:
            json.dump(annotations, output_file, indent=4)
        upload_if_needed(output_path, output_uri)


if __name__ == '__main__':
    merge_predictions()
