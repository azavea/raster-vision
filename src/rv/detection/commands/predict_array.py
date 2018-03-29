import os
import json

import click

from rv.detection.commands.settings import (
    temp_root_dir, default_channel_order)
from rv.utils.files import (
    download_if_needed, MyTemporaryDirectory)
from rv.detection.commands.predict import _predict


@click.command()
@click.argument('inference_graph_uri')
@click.argument('label_map_uri')
@click.argument('projects_uri')
@click.argument('output_dir_uri')
@click.option('--mask-uri', default=None,
              help='URI for mask GeoJSON file to use as filter for detections')
@click.option('--channel-order', nargs=3, type=int,
              default=default_channel_order, help='Index of RGB channels')
@click.option('--chip-size', default=300)
@click.option('--score-thresh', default=0.5,
              help='Score threshold of predictions to keep')
@click.option('--merge-thresh', default=0.5,
              help='IOU threshold for merging predictions')
@click.option('--save-temp', is_flag=True)
def predict_array(inference_graph_uri, label_map_uri, projects_uri,
                  output_dir_uri, mask_uri, channel_order, chip_size,
                  score_thresh, merge_thresh, save_temp):
    job_ind = int(os.environ['AWS_BATCH_JOB_ARRAY_INDEX'])

    prefix = temp_root_dir
    temp_dir = os.path.join(prefix, 'predict-array') if save_temp else None
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        projects_path = download_if_needed(projects_uri, temp_dir)
        with open(projects_path, 'r') as projects_file:
            projects = json.load(projects_file)

            def get_image_coords():
                image_ind = 0
                for project_ind, project in enumerate(projects):
                    for project_image_ind, image_uri in enumerate(project['images']):
                        if job_ind == image_ind:
                            return project_ind, project_image_ind
                        image_ind += 1

            # Run predict for single image and generate ouput_uri based on
            # the index of the image within the project.
            project_ind, project_image_ind = get_image_coords()
            project = projects[project_ind]
            image_uri = project['images'][project_image_ind]
            image_uris = [image_uri]
            output_uri = os.path.join(
                output_dir_uri, project['id'],
                '{}.json'.format(project_image_ind))
            output_debug_uri = None

            _predict(inference_graph_uri, label_map_uri, image_uris,
                     output_uri, output_debug_uri, mask_uri,
                     channel_order, chip_size, score_thresh, merge_thresh,
                     save_temp)


if __name__ == '__main__':
    predict_array()
