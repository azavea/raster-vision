import os
import json
import copy

import click

from rv.detection.commands.settings import (
    temp_root_dir)
from rv.utils.files import (
    download_if_needed, get_local_path, upload_if_needed,
    MyTemporaryDirectory)


def _merge_predictions(predictions_list):
    merged_predictions = copy.deepcopy(predictions_list[0])
    for predictions in predictions_list[1:]:
        merged_predictions['features'].extend(predictions['features'])
    return merged_predictions


@click.command()
@click.argument('projects_uri')
@click.argument('output_dir_uri')
@click.option('--save-temp', is_flag=True)
def merge_predictions(projects_uri, output_dir_uri, save_temp):
    prefix = temp_root_dir
    temp_dir = os.path.join(prefix, 'merge-predictions') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        projects_path = download_if_needed(projects_uri, temp_dir)

        # For each project:
        # download the predictions files, merge them, and upload the merged
        # predictions.
        projects = json.load(open(projects_path))
        for project in projects:
            predictions_list = []
            for image_ind, image in enumerate(project['images']):
                predictions_uri = os.path.join(
                    output_dir_uri, project['id'],
                    '{}.json'.format(image_ind))
                predictions_path = download_if_needed(
                    predictions_uri, temp_dir)
                predictions_list.append(json.load(open(predictions_path)))

            output_uri = project['annotations']
            output_path = get_local_path(output_uri, temp_dir)
            predictions = _merge_predictions(predictions_list)
            json.dump(predictions, open(output_path, 'w'))
            upload_if_needed(output_path, output_uri)


if __name__ == '__main__':
    merge_predictions()
