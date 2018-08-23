#!/usr/bin/env python

import tempfile
import json
import os
import math
import traceback

import click
import numpy as np
import tensorflow

import rastervision.workflows.chain as chain_workflow

CLASSIFICATION = 'classification'
OBJECT_DETECTION = 'object-detection'
all_tests = [CLASSIFICATION, OBJECT_DETECTION]

np.random.seed(1234)
tensorflow.set_random_seed(5678)

# Suppress warnings and info to avoid cluttering CI log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TEST_ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../fixtures/tests')


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
    return os.path.join(TEST_ROOT_DIR, test)


def get_workflow_path(test):
    return os.path.join(get_test_dir(test), 'configs/workflow.json')


def get_workflow_copy_path(test, temp_dir):
    return os.path.join(temp_dir, '{}-workflow.json'.format(test))


def get_expected_eval_path(test):
    return os.path.join(get_test_dir(test), 'expected-output/eval.json')


def get_actual_eval_path(test, temp_dir):
    return os.path.join(
        temp_dir, ('rv-output/raw-datasets/{}-test/datasets/' +
                   'default/models/default/predictions/default/evals/default/'
                   + 'output/eval.json').format(test))


def open_json(path):
    with open(path, 'r') as file:
        return json.load(file)


def setup_test(test, temp_dir):
    # Copy workflow config and set local_uri_map.
    workflow_path = get_workflow_path(test)
    workflow_copy_path = get_workflow_copy_path(test, temp_dir)

    workflow_dict = open_json(workflow_path)
    workflow_dict['local_uri_map']['rv_root'] = temp_dir
    workflow_dict['local_uri_map']['test_dir'] = get_test_dir(test)

    with open(workflow_copy_path, 'w') as workflow_file:
        json.dump(workflow_dict, workflow_file)

    return workflow_copy_path


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


def run_test(test, temp_dir):
    errors = []
    tasks = []
    workflow_path = setup_test(test, temp_dir)

    # Check that running workflow doesn't raise any exceptions.
    try:
        chain_workflow._main(workflow_path, tasks, run=True)
    except Exception as exc:
        errors.append(
            TestError(test, 'raised an exception while running',
                      traceback.format_exc()))
        return errors

    # Check that the eval is similar to expected eval.
    errors.extend(check_eval(test, temp_dir))

    return errors


@click.command()
@click.argument('tests', nargs=-1)
def main(tests):
    """Runs RV end-to-end and checks that evaluation metrics are correct."""
    if len(tests) == 0:
        tests = all_tests

    with tempfile.TemporaryDirectory() as temp_dir:
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
