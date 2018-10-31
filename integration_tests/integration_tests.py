#!/usr/bin/env python

from copy import deepcopy
import json
import os
import math
import traceback

import click
import numpy as np
import tensorflow

import rastervision as rv

from integration_tests.chip_classification_tests.experiment \
    import ChipClassificationIntegrationTest
from integration_tests.object_detection_tests.experiment \
    import ObjectDetectionIntegrationTest
from integration_tests.semantic_segmentation_tests.experiment \
    import SemanticSegmentationIntegrationTest
from rastervision.rv_config import RVConfig

all_tests = [
    rv.CHIP_CLASSIFICATION, rv.OBJECT_DETECTION, rv.SEMANTIC_SEGMENTATION
]

np.random.seed(1234)
tensorflow.set_random_seed(5678)

# Suppress warnings and info to avoid cluttering CI log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TEST_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class IntegrationTestExperimentRunner(rv.runner.LocalExperimentRunner):
    def __init__(self, tmp_dir=None):
        super().__init__(tmp_dir)

    def _run_experiment(self, command_dag):
        """Check serialization of all commands."""
        for command_config in command_dag.get_sorted_commands():
            deepcopy(
                rv.command.CommandConfig.from_proto(command_config.to_proto()))

        super()._run_experiment(command_dag)


def console_info(msg):
    click.echo(click.style(msg, fg='green'))


def console_warning(msg):
    click.echo(click.style(msg, fg='yellow'))


def console_error(msg):
    click.echo(click.style(msg, fg='red', err=True))


class TestError():
    def __init__(self, test, message, details=None):
        self.test = test
        self.message = message
        self.details = details

    def __str__(self):
        return ('Error\n' + '------\n' + 'Test: {}\n'.format(self.test) +
                'Message: {}\n'.format(self.message) + 'Details: {}'.format(
                    str(self.details)) if self.details else '' + '\n')


def get_test_dir(test):
    return os.path.join(TEST_ROOT_DIR, test.lower().replace('-', '_'))


def get_expected_eval_path(test):
    return os.path.join('{}_tests'.format(get_test_dir(test)),
                        'expected-output/eval.json')


def get_actual_eval_path(test, temp_dir):
    return os.path.join(temp_dir, test.lower(), 'eval/default/eval.json')


def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def check_eval_item(test, expected_item, actual_item):
    errors = []
    f1_threshold = 0.01
    class_name = expected_item['class_name']

    expected_f1 = expected_item['f1'] or 0.0
    actual_f1 = actual_item['f1'] or 0.0
    if math.fabs(expected_f1 - actual_f1) > f1_threshold:
        errors.append(
            TestError(
                test, 'F1 scores are not close enough',
                'for class_name: {} expected f1: {}, actual f1: {}'.format(
                    class_name, expected_item['f1'], actual_item['f1'])))

    return errors


def check_eval(test, temp_dir):
    errors = []

    actual_eval_path = get_actual_eval_path(test, temp_dir)
    expected_eval_path = get_expected_eval_path(test)

    if os.path.isfile(actual_eval_path):
        expected_eval = open_json(expected_eval_path)
        actual_eval = open_json(actual_eval_path)

        for expected_item in expected_eval:
            class_name = expected_item['class_name']
            actual_item = \
                next(filter(
                    lambda x: x['class_name'] == class_name, actual_eval))
            errors.extend(check_eval_item(test, expected_item, actual_item))
    else:
        errors.append(
            TestError(test, 'actual eval file does not exist',
                      actual_eval_path))

    return errors


def get_experiment(test, tmp_dir):
    if test == rv.OBJECT_DETECTION:
        return ObjectDetectionIntegrationTest().exp_main(
            os.path.join(tmp_dir, test.lower()))
    if test == rv.CHIP_CLASSIFICATION:
        return ChipClassificationIntegrationTest().exp_main(
            os.path.join(tmp_dir, test.lower()))
    if test == rv.SEMANTIC_SEGMENTATION:
        return SemanticSegmentationIntegrationTest().exp_main(
            os.path.join(tmp_dir, test.lower()))

    raise Exception('Unknown test {}'.format(test))


def run_test(test, temp_dir):
    errors = []
    experiment = get_experiment(test, temp_dir)

    # Check serialization
    msg = experiment.to_proto()
    experiment = rv.ExperimentConfig.from_proto(msg)

    # Check that running doesn't raise any exceptions.
    try:
        IntegrationTestExperimentRunner(os.path.join(temp_dir, test.lower())) \
            .run(experiment, rerun_commands=True)

    except Exception as exc:
        errors.append(
            TestError(test, 'raised an exception while running',
                      traceback.format_exc()))
        return errors

    # Check that the eval is similar to expected eval.
    errors.extend(check_eval(test, temp_dir))

    if not errors:
        # Check the prediction package
        # This will only work with raster_sources that
        # have a single URI.
        skip = False

        experiment = experiment.fully_resolve()

        scenes_to_uris = {}
        scenes = experiment.dataset.validation_scenes
        for scene in scenes:
            rs = scene.raster_source
            if hasattr(rs, 'uri'):
                scenes_to_uris[scene.id] = rs.uri
            elif hasattr(rs, 'uris'):
                uris = rs.uris
                if len(uris) > 1:
                    skip = True
                else:
                    scenes_to_uris[scene.id] = uris[0]
            else:
                skip = True

        if skip:
            console_warning('Skipping predict package test for '
                            'test {}, experiment {}'.format(
                                test, experiment.id))
        else:
            console_info('Checking predict package produces same results...')
            pp = experiment.task.predict_package_uri
            predict = rv.Predictor(pp, temp_dir).predict

            for scene_config in scenes:
                # Need to write out labels and read them back,
                # otherwise the floating point precision direct box
                # coordinates will not match those from the PREDICT
                # command, which are rounded to pixel coordinates
                # via pyproj logic (in the case of rasterio crs transformer.
                predictor_label_store_uri = os.path.join(
                    temp_dir, test.lower(),
                    'predictor/{}'.format(scene_config.id))
                uri = scenes_to_uris[scene_config.id]
                predict(uri, predictor_label_store_uri)
                scene = scene_config.create_scene(experiment.task, temp_dir)
                scene_labels = scene.prediction_label_store.get_labels()

                extent = scene.raster_source.get_extent()
                crs_transformer = scene.raster_source.get_crs_transformer()
                predictor_labels = scene_config.label_store \
                                               .for_prediction(
                                                   predictor_label_store_uri) \
                                               .create_store(
                                                   experiment.task,
                                                   extent,
                                                   crs_transformer,
                                                   temp_dir) \
                                               .get_labels()

                if not predictor_labels == scene_labels:
                    e = TestError(
                        test, ('Predictor did not produce the same labels '
                               'as the Predict command'),
                        'for scene {} in experiment {}'.format(
                            scene_config.id, experiment.id))
                    errors.append(e)

    return errors


@click.command()
@click.argument('tests', nargs=-1)
def main(tests):
    """Runs RV end-to-end and checks that evaluation metrics are correct."""
    if len(tests) == 0:
        tests = all_tests

    tests = list(map(lambda x: x.upper(), tests))

    with RVConfig.get_tmp_dir() as temp_dir:
        errors = []
        for test in tests:
            if test not in all_tests:
                print('{} is not a valid test.'.format(test))
                return

            errors.extend(run_test(test, temp_dir))

            for error in errors:
                print(error)

        for test in tests:
            nb_test_errors = len(
                list(filter(lambda error: error.test == test, errors)))
            if nb_test_errors == 0:
                print('{} test passed!'.format(test))

        if errors:
            exit(1)


if __name__ == '__main__':
    main()
