import json
import os
import click

from rv.utils.files import (
    download_if_needed, MyTemporaryDirectory)
from rv.utils.batch import _batch_submit
from rv.detection.commands.settings import temp_root_dir


def make_predict_array_cmd(inference_graph_uri, label_map_uri, projects_uri,
                           output_dir_uri):
    return 'python -m rv.detection.run predict_array {} {} {} {}'.format(
        inference_graph_uri, label_map_uri, projects_uri, output_dir_uri)


def make_merge_predictions_cmd(projects_uri, output_dir_uri):
    return 'python -m rv.detection.run merge_predictions {} {}'.format(
        projects_uri, output_dir_uri)


def get_nb_images(projects):
    nb_images = 0
    for project in projects:
        nb_images += len(project['images'])
    return nb_images


@click.command()
@click.argument('projects_uri')
@click.argument('label_map_uri')
@click.argument('inference_graph_uri')
@click.argument('output_dir_uri')
@click.option('--branch-name', default='develop')
@click.option('--attempts', default=1)
@click.option('--cpu', is_flag=True)
def parallel_predict(projects_uri, label_map_uri, inference_graph_uri,
                     output_dir_uri,
                     branch_name, attempts, cpu):
    prefix = temp_root_dir
    temp_dir = os.path.join(prefix, 'parallel-predict')
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        # Load projects and count number of images
        projects_path = download_if_needed(projects_uri, temp_dir)
        projects = json.load(open(projects_path))
        nb_images = get_nb_images(projects)

        # Submit an array job with nb_images elements.
        command = make_predict_array_cmd(
            inference_graph_uri, label_map_uri, projects_uri, output_dir_uri)
        '''
        predict_job_id = _batch_submit(
            branch_name, command, attempts=attempts, cpu=cpu,
            array_size=nb_images)
        '''
        # Submit a dependent merge_predictions job.
        command = make_merge_predictions_cmd(
            projects_uri, output_dir_uri)
        _batch_submit(
            branch_name, command, attempts=attempts, cpu=cpu)
            #parent_job_ids=[predict_job_id])


if __name__ == '__main__':
    parallel_predict()
