import json
import os

from rv.utils.files import (
    download_if_needed, MyTemporaryDirectory)
from rv.utils.batch import _batch_submit
from rv.detection.commands.settings import temp_root_dir


def make_predict_array_cmd(inference_graph_uri, label_map_uri, projects_uri):
    return 'python -m rv.detection.run predict_array {} {} {}'.format(
        inference_graph_uri, label_map_uri, projects_uri)


def make_merge_predictions_cmd(projects_uri, output_uri):
    return 'python -m rv.detection.run merge_predictions {} {}'.format(
        projects_uri, output_uri)


def parallel_predict():
    inference_graph_uri = \
        's3://raster-vision-lf-dev/detection/trained-models/cowc-potsdam/30cm/inference-graph.pb'
    label_map_uri = \
        's3://raster-vision-lf-dev/detection/configs/label-maps/cowc.pbtxt'
    projects_uri = \
        's3://raster-vision-lf-dev/detection/configs/projects/predict/cowc-potsdam/remote/30cm-test.json'
    output_uri = \
        's3://raster-vision-lf-dev/detection/predictions/cowc-potsdam/30cm-test/all.json'

    branch_name = 'lf/rfint'
    attempts = 1
    cpu = True

    prefix = temp_root_dir
    temp_dir = os.path.join(prefix, 'parallel-predict')
    with MyTemporaryDirectory(temp_dir, prefix) as temp_dir:
        projects_path = download_if_needed(projects_uri, temp_dir)
        with open(projects_path, 'r') as projects_file:
            projects = json.load(projects_file)
            nb_projects = len(projects)
            command = make_predict_array_cmd(
                inference_graph_uri, label_map_uri, projects_uri)
            predict_job_id = _batch_submit(
                branch_name, command, attempts=attempts, cpu=cpu,
                array_size=nb_projects)

        command = make_merge_predictions_cmd(
            projects_uri, output_uri)
        _batch_submit(
            branch_name, command, attempts=attempts, cpu=cpu,
            parent_job_ids=[predict_job_id])


if __name__ == '__main__':
    parallel_predict()
