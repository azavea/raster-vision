#!/usr/bin/env python

from typing import List
from os.path import join, dirname, abspath, isfile
import math
import traceback
import importlib
from pprint import pformat

import click
import numpy as np

from rastervision.pipeline import rv_config, Verbosity
from rastervision.pipeline.file_system import file_to_json
from rastervision.pipeline.runner import InProcessRunner
from rastervision.pipeline.cli import _run_pipeline
from rastervision.core import Predictor

ALL_TESTS = {
    'chip_classification.basic': {
        'task': 'chip_classification',
        'module': 'integration_tests.chip_classification',
        'kwargs': {
            'nochip': False
        }
    },
    'chip_classification.nochip': {
        'task': 'chip_classification',
        'module': 'integration_tests.chip_classification',
        'kwargs': {
            'nochip': True
        }
    },
    'object_detection.basic': {
        'task': 'object_detection',
        'module': 'integration_tests.object_detection',
        'kwargs': {
            'nochip': False
        }
    },
    'object_detection.nochip': {
        'task': 'object_detection',
        'module': 'integration_tests.object_detection',
        'kwargs': {
            'nochip': True
        }
    },
    'semantic_segmentation.basic': {
        'task': 'semantic_segmentation',
        'module': 'integration_tests.semantic_segmentation',
        'kwargs': {
            'nochip': False
        }
    },
    'semantic_segmentation.nochip': {
        'task': 'semantic_segmentation',
        'module': 'integration_tests.semantic_segmentation',
        'kwargs': {
            'nochip': True
        }
    }
}
TEST_ROOT_DIR = dirname(abspath(__file__))

np.random.seed(1234)


def console_info(msg: str, **kwargs) -> None:
    click.secho(msg, fg='magenta', **kwargs)


def console_warning(msg: str, **kwargs) -> None:
    click.secho(msg, fg='yellow', **kwargs)


def console_error(msg: str, **kwargs) -> None:
    click.secho(msg, fg='red', err=True, **kwargs)


def console_success(msg: str, **kwargs) -> None:
    click.secho(msg, fg='cyan', **kwargs)


class TestError():
    def __init__(self, test, message, details=None):
        self.test = test
        self.message = message
        self.details = details

    def __str__(self):
        return ('Error\n'
                '------\n'
                f'Test: {self.test}\n'
                f'Message: {self.message}\n'
                f'Details: {str(self.details)}' if self.details else '\n')


def get_test_dir(test_id: str) -> str:
    return join(TEST_ROOT_DIR, test_id.replace('-', '_'))


def get_expected_eval_path(test_id: str, test_cfg: dict) -> str:
    return join(get_test_dir(test_cfg['task']), 'expected-output/eval.json')


def get_actual_eval_path(test_id: str, tmp_dir: str) -> str:
    return join(tmp_dir, test_id, 'eval/validation_scenes/eval.json')


def check_eval_item(test_id: str, test_cfg: dict, expected_item: dict,
                    actual_item: dict) -> List[TestError]:
    errors = []
    f1_threshold = 0.05
    class_name = expected_item['class_name']

    expected_f1 = expected_item['metrics']['f1'] or 0.0
    actual_f1 = actual_item['metrics']['f1'] or 0.0
    if math.fabs(expected_f1 - actual_f1) > f1_threshold:
        errors.append(
            TestError(
                test_id, 'F1 scores are not close enough',
                f'for class "{class_name}": '
                f'expected f1: {expected_f1}, actual f1: {actual_f1}'))

    return errors


def check_eval(test_id: str, test_cfg: dict, tmp_dir: str) -> List[TestError]:
    errors = []

    actual_eval_path = get_actual_eval_path(test_id, tmp_dir)
    expected_eval_path = get_expected_eval_path(test_id, test_cfg)

    if isfile(actual_eval_path):
        expected_eval = file_to_json(expected_eval_path)['overall']
        actual_eval = file_to_json(actual_eval_path)['overall']

        for expected_item in expected_eval:
            class_name = expected_item['class_name']
            actual_item = \
                next(filter(
                    lambda x: x['class_name'] == class_name, actual_eval))
            errors.extend(
                check_eval_item(test_id, test_cfg, expected_item, actual_item))
    else:
        errors.append(
            TestError(test_id, 'actual eval file does not exist',
                      actual_eval_path))

    return errors


def test_model_bundle_validation(pipeline, test_id: str, test_cfg: dict,
                                 tmp_dir: str,
                                 image_uri: str) -> List[TestError]:
    console_info('Checking predict command validation...')
    errors = []
    model_bundle_uri = pipeline.get_model_bundle_uri()
    predictor = Predictor(model_bundle_uri, tmp_dir, channel_order=[0, 1, 7])
    try:
        predictor.predict([image_uri], 'x.txt')
        e = TestError(test_id,
                      ('Predictor should have raised exception due to invalid '
                       'channel_order, but did not.'))
        errors.append(e)
    except ValueError:
        pass

    return errors


def test_model_bundle_results(pipeline, test_id: str, test_cfg: dict,
                              tmp_dir: str, scenes: list,
                              scenes_to_uris: dict) -> List[TestError]:
    console_info('Checking model bundle produces same results...')
    errors = []
    model_bundle_uri = pipeline.get_model_bundle_uri()
    predictor = Predictor(model_bundle_uri, tmp_dir)

    for scene_cfg in scenes:
        # Need to write out labels and read them back,
        # otherwise the floating point precision direct box
        # coordinates will not match those from the PREDICT
        # command, which are rounded to pixel coordinates
        # via pyproj logic (in the case of rasterio crs transformer.
        scene = scene_cfg.build(pipeline.dataset.class_config, tmp_dir)

        predictor_label_store_uri = join(tmp_dir, test_id.lower(),
                                         'predictor/{}'.format(scene_cfg.id))
        image_uri = scenes_to_uris[scene_cfg.id]
        predictor.predict([image_uri], predictor_label_store_uri)

        extent = scene.raster_source.get_extent()
        crs_transformer = scene.raster_source.get_crs_transformer()
        predictor_label_store = scene_cfg.label_store.copy()
        predictor_label_store.uri = predictor_label_store_uri
        predictor_label_store = predictor_label_store.build(
            pipeline.dataset.class_config, crs_transformer, extent, tmp_dir)

        from rastervision.core.data import ActivateMixin
        with ActivateMixin.compose(scene, predictor_label_store):
            bundle_labels = predictor_label_store.get_labels()
            predict_stage_labels = scene.label_store.get_labels()
            if bundle_labels != predict_stage_labels:
                e = TestError(test_id,
                              ('Predictor did not produce the same labels '
                               'as the Predict command'),
                              'for scene {}'.format(scene_cfg.id))
                errors.append(e)

    return errors


def test_model_bundle(pipeline,
                      test_id: str,
                      test_cfg: dict,
                      tmp_dir: str,
                      check_channel_order: bool = False) -> List[TestError]:
    # Check the model bundle.
    # This will only work with raster_sources that
    # have a single URI.
    skip = False
    errors = []

    scenes_to_uris = {}
    scenes = pipeline.dataset.validation_scenes
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
        console_warning('Skipping predict package test for test {}.')
    else:
        if check_channel_order:
            errors.extend(
                test_model_bundle_validation(pipeline, test_id, test_cfg,
                                             tmp_dir, uris[0]))
        else:
            errors.extend(
                test_model_bundle_results(pipeline, test_id, test_cfg, tmp_dir,
                                          scenes, scenes_to_uris))

    return errors


def run_test(test_id: str, test_cfg: dict, tmp_dir: str) -> List[TestError]:
    msg = f'\nRunning test: {test_id}'
    console_info(msg, bold=True)
    console_info('With params:')
    console_info(pformat(test_cfg))

    errors = []
    root_uri = join(tmp_dir, test_id)
    runner = 'inprocess'
    config_mod = importlib.import_module(f'{test_cfg["module"]}.config')
    pipeline_cfg = config_mod.get_config(runner, root_uri,
                                         **test_cfg['kwargs'])
    pipeline_cfg.update()
    runner = InProcessRunner()

    # Check that running doesn't raise any exceptions.
    try:
        _run_pipeline(pipeline_cfg, runner, tmp_dir)
    except Exception:
        errors.append(
            TestError(test_id, 'raised an exception while running',
                      traceback.format_exc()))
        return errors

    # Check that the eval is similar to expected eval.
    errors.extend(check_eval(test_id, test_cfg, tmp_dir))

    if not errors:
        errors.extend(
            test_model_bundle(pipeline_cfg, test_id, test_cfg, tmp_dir))
        errors.extend(
            test_model_bundle(
                pipeline_cfg,
                test_id,
                test_cfg,
                tmp_dir,
                check_channel_order=True))
    return errors


@click.command()
@click.argument('tests', nargs=-1)
@click.option(
    '--root-uri',
    '-t',
    help=('Sets the rv_root directory used. '
          'If set, test will not clean this directory up.'))
@click.option(
    '--verbose', '-v', is_flag=True, help=('Sets the logging level to DEBUG.'))
def main(tests, root_uri, verbose):
    """Runs RV end-to-end and checks that evaluation metrics are correct."""
    if verbose:
        rv_config.set_verbosity(verbosity=Verbosity.DEBUG)

    if len(tests) == 0:
        # no tests specified, so run all
        tests = list(ALL_TESTS.keys())
    else:
        # run all tests that start with the given string e.g "chip" will match
        # both "chip_classification.basic" and "chip_classification.nochip"
        _tests = []
        for t in tests:
            t = t.strip().lower()
            matching_tests = [k for k in ALL_TESTS.keys() if k.startswith(t)]
            _tests.extend(matching_tests)
            if len(matching_tests) == 0:
                console_error(
                    f'{t} does not match any valid tests. Valid tests are: ')
                console_error(pformat(list(ALL_TESTS.keys())))
                continue
        tests = _tests

    console_info('The following tests will be run:')
    console_info(pformat(tests, compact=False))

    with rv_config.get_tmp_dir() as tmp_dir:
        if root_uri:
            tmp_dir = root_uri

        num_failed = 0
        errors = {}
        for test_id in tests:
            test_cfg = ALL_TESTS[test_id]
            errors[test_id] = run_test(test_id, test_cfg, tmp_dir)
            if len(errors[test_id]) > 0:
                num_failed += 1

            for error in errors[test_id]:
                console_error(str(error))

        for test_id in tests:
            if test_id not in errors:
                continue
            if len(errors[test_id]) == 0:
                console_success(f'{test_id}: test passed!', bold=True)
            else:
                console_error(f'{test_id}: test failed!', bold=True)

        if num_failed > 0:
            console_error(
                f'Tests passed: {len(tests) - num_failed} of {len(tests)}')
            console_error('Error counts:')
            console_error(pformat({k: len(es) for k, es in errors.items()}))
            exit(1)


if __name__ == '__main__':
    main()
