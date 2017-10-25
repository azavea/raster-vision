from os.path import join, dirname
from os import makedirs
import json

import click

from rv.commands.predict import _predict
from rv.commands.eval_predictions import _eval_predictions
from rv.commands.utils import (
    download_if_needed, get_local_path, upload_if_needed, load_projects,
    make_temp_dir)
from rv.commands.settings import planet_channel_order, temp_root_dir


@click.command()
@click.argument('inference_graph_uri')
@click.argument('projects_uri')
@click.argument('label_map_uri')
@click.argument('output_uri')
@click.option('--chip-size', default=300, help='Height and width of each chip')
@click.option('--channel-order', nargs=3, type=int,
              default=planet_channel_order, help='Indices of the RGB channels')
def eval_model(inference_graph_uri, projects_uri, label_map_uri, output_uri,
               chip_size, channel_order):
    """Evaluate a model on a set of projects with ground truth annotations.

    Makes predictions using a model on a set of projects and then compares them
    with ground truth annotations, saving the average precision and recall
    across the projects.

    Args:
        inference_graph_uri: the inference graph of the model to evaluate
        projects_uri: the JSON file with the images and annotations for a
            set of projects
        label_map_uri: label map for the model
        output_uri: the destination for the JSON output
    """
    temp_dir = join(temp_root_dir, 'eval_projects')
    make_temp_dir(temp_dir)
    predictions_dir = join(temp_dir, 'predictions')
    makedirs(predictions_dir, exist_ok=True)
    evals_dir = join(temp_dir, 'eval')
    makedirs(evals_dir, exist_ok=True)

    projects_path = download_if_needed(temp_dir, projects_uri)
    image_paths_list, annotations_paths = \
        load_projects(temp_dir, projects_path)

    output_path = get_local_path(temp_dir, output_uri)
    output_dir = dirname(output_path)
    makedirs(output_dir, exist_ok=True)

    # Run prediction and evaluation on each project.
    eval_paths = []
    for project_ind, (image_paths, annotations_path) in \
            enumerate(zip(image_paths_list, annotations_paths)):
        predictions_path = join(predictions_dir, '{}.json'.format(project_ind))
        eval_path = join(evals_dir, '{}.json'.format(project_ind))
        eval_paths.append(eval_path)
        _predict(inference_graph_uri, label_map_uri, image_paths,
                 predictions_path, channel_order=channel_order,
                 chip_size=chip_size)
        _eval_predictions(
            image_paths, label_map_uri, annotations_path, predictions_path,
            eval_path)

    # Average evals and save.
    precision_sums = {}
    recall_sums = {}
    for eval_path in eval_paths:
        with open(eval_path, 'r') as eval_file:
            project_eval = json.load(eval_file)
            for label_eval in project_eval:
                name = label_eval['name']
                precision_sums[name] = \
                    precision_sums.get(name, 0) + label_eval['precision']
                recall_sums[name] = \
                    recall_sums.get(name, 0) + label_eval['recall']

    avg_evals = []
    nb_projects = len(eval_paths)
    for name in precision_sums.keys():
        avg_evals.append({
            'name': name,
            'avg_precision': precision_sums[name] / nb_projects,
            'avg_recall': recall_sums[name] / nb_projects
        })

    with open(output_path, 'w') as output_file:
        json.dump(avg_evals, output_file, indent=4)
    upload_if_needed(output_path, output_uri)


if __name__ == '__main__':
    eval_model()
