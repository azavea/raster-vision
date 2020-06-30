# flake8: noqa

import pprint
import subprocess
from os.path import join, basename, dirname

import click

from rastervision.pipeline.file_system import (
    list_paths, download_or_copy, file_exists, make_dir, file_to_json)

cfg = [
    {
        'key':
        'spacenet-rio-cc',
        'module':
        'rastervision.pytorch_backend.examples.chip_classification.spacenet_rio',
        'local': {
            'raw_uri': '/opt/data/raw-data/spacenet-dataset',
            'processed_uri': '/opt/data/examples/spacenet/rio/processed-data',
            'root_uri': '/opt/data/examples/spacenet-rio-cc'
        },
        'remote': {
            'raw_uri':
            's3://spacenet-dataset/',
            'processed_uri':
            's3://raster-vision-lf-dev/examples/spacenet/rio/processed-data',
            'root_uri':
            's3://raster-vision-lf-dev/examples/spacenet-rio-cc'
        },
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/spacenet-rio-cc/013022223130_sample.tif'
    },
    {
        'key':
        'isprs-potsdam-ss',
        'module':
        'rastervision.pytorch_backend.examples.semantic_segmentation.isprs_potsdam',
        'local': {
            'raw_uri': '/opt/data/raw-data/isprs-potsdam/',
            'processed_uri': '/opt/data/examples/potsdam/processed-data',
            'root_uri': '/opt/data/examples/isprs-potsdam-ss/'
        },
        'remote': {
            'raw_uri':
            's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri':
            's3://raster-vision-lf-dev/examples/potsdam/processed-data',
            'root_uri':
            's3://raster-vision-lf-dev/examples/isprs-potsdam-ss'
        },
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/isprs-potsdam-ss/3_12_sample.tif'
    },
    {
        'key':
        'spacenet-vegas-buildings-ss',
        'module':
        'rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_vegas',
        'local': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': '/opt/data/examples/spacenet-vegas-buildings-ss'
        },
        'remote': {
            'raw_uri':
            's3://spacenet-dataset/',
            'root_uri':
            's3://raster-vision-lf-dev/examples/spacenet-vegas-buildings-ss'
        },
        'extra_args': [['target', 'buildings']],
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/spacenet-vegas-buildings-ss/1929.tif'
    },
    {
        'key':
        'spacenet-vegas-roads-ss',
        'module':
        'rastervision.pytorch_backend.examples.semantic_segmentation.spacenet_vegas',
        'local': {
            'raw_uri': 's3://spacenet-dataset/',
            'root_uri': '/opt/data/examples/spacenet-vegas-roads-ss'
        },
        'remote': {
            'raw_uri':
            's3://spacenet-dataset/',
            'root_uri':
            's3://raster-vision-lf-dev/examples/spacenet-vegas-roads-ss'
        },
        'extra_args': [['target', 'roads']],
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/spacenet-vegas-roads-ss/524.tif'
    },
    {
        'key':
        'cowc-potsdam-od',
        'module':
        'rastervision.pytorch_backend.examples.object_detection.cowc_potsdam',
        'local': {
            'raw_uri': '/opt/data/raw-data/isprs-potsdam',
            'processed_uri': '/opt/data/examples/cowc-potsdam/processed-data',
            'root_uri': '/opt/data/examples/cowc-potsdam-od'
        },
        'remote': {
            'raw_uri':
            's3://raster-vision-raw-data/isprs-potsdam',
            'processed_uri':
            's3://raster-vision-lf-dev/examples/cowc-potsdam/processed-data',
            'root_uri':
            's3://raster-vision-lf-dev/examples/cowc-potsdam-od'
        },
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/cowc-potsdam-od/3_10_sample.tif'
    },
    {
        'key':
        'xview-od',
        'module':
        'rastervision.pytorch_backend.examples.object_detection.xview',
        'local': {
            'raw_uri': 's3://raster-vision-xview-example/raw-data',
            'processed_uri': '/opt/data/examples/xview/processed-data',
            'root_uri': '/opt/data/examples/xview-od'
        },
        'remote': {
            'raw_uri':
            's3://raster-vision-xview-example/raw-data',
            'processed_uri':
            's3://raster-vision-lf-dev/examples/xview/processed-data',
            'root_uri':
            's3://raster-vision-lf-dev/examples/xview-od'
        },
        'sample_img':
        'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/xview-od/1124-sample.tif'
    },
]


def validate_keys(keys):
    exp_keys = [exp_cfg['key'] for exp_cfg in cfg]
    invalid_keys = set(keys).difference(exp_keys)
    if invalid_keys:
        raise ValueError('{} are invalid keys'.format(', '.join(invalid_keys)))


def _run(exp_cfg, output_dir, test=True, remote=False, commands=None):
    uris = exp_cfg['remote'] if remote else exp_cfg['local']
    cmd = ['rastervision']
    rv_profile = exp_cfg.get('rv_profile')
    if rv_profile is not None:
        cmd += ['-p', rv_profile]
    cmd += ['run', 'batch' if remote else 'inprocess', exp_cfg['module']]
    if commands is not None:
        cmd += commands
    cmd += ['-a', 'raw_uri', uris['raw_uri']]
    if 'processed_uri' in uris:
        cmd += ['-a', 'processed_uri', uris['processed_uri']]
    root_uri = join(uris['root_uri'], output_dir)
    cmd += ['-a', 'root_uri', root_uri]
    cmd += ['-a', 'test', 'True' if test else 'False']
    extra_args = exp_cfg.get('extra_args')
    if extra_args:
        for k, v in extra_args:
            cmd += ['-a', str(k), str(v)]
    if remote:
        cmd += ['--splits', '3']

    print('running command:')
    print(' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('failure!')
        print(' '.join(cmd))
        exit()


@click.group()
def test():
    pass


@test.command()
@click.argument('output_dir')
@click.argument('keys', nargs=-1)
@click.option('--test', is_flag=True)
@click.option('--remote', is_flag=True)
@click.option('--commands')
def run(output_dir, keys, test, remote, commands):
    run_all = len(keys) == 0
    validate_keys(keys)

    if commands is not None:
        commands = commands.split(' ')
    for exp_cfg in cfg:
        if run_all or exp_cfg['key'] in keys:
            _run(
                exp_cfg,
                output_dir,
                test=test,
                remote=remote,
                commands=commands)


def _collect(key, root_uri, output_dir, collect_dir, get_model_bundle=False):
    print('\nCollecting experiment {}...\n'.format(key))

    model_bundle_uri = join(root_uri, output_dir, 'bundle', 'model-bundle.zip')
    eval_uri = join(root_uri, output_dir, 'eval', 'eval.json')

    if not file_exists(eval_uri):
        print('Missing eval!')
        return

    if not file_exists(model_bundle_uri):
        print('Missing model bundle!')
        return

    make_dir(join(collect_dir, key))
    if get_model_bundle:
        download_or_copy(model_bundle_uri, join(collect_dir, key))

    download_or_copy(eval_uri, join(collect_dir, key))

    eval_json = file_to_json(join(collect_dir, key, 'eval.json'))
    pprint.pprint(eval_json['overall'], indent=4)


@test.command()
@click.argument('output_dir')
@click.argument('collect_dir')
@click.argument('keys', nargs=-1)
@click.option('--remote', is_flag=True)
@click.option('--get-model-bundle', is_flag=True)
def collect(output_dir, collect_dir, keys, remote, get_model_bundle):
    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        if run_all or exp_cfg['key'] in keys:
            key = exp_cfg['key']
            uris = exp_cfg['remote'] if remote else exp_cfg['local']
            root_uri = uris['root_uri']
            _collect(key, root_uri, output_dir, collect_dir, get_model_bundle)


def _predict(key, sample_img, collect_dir):
    print('\nTesting model bundle for {}...\n'.format(key))

    model_bundle_uri = join(collect_dir, key, 'model-bundle.zip')
    if not file_exists(model_bundle_uri):
        print('Bundle does not exist!')
        return

    sample_img = download_or_copy(sample_img, join(collect_dir, key))

    exts = {'cc': 'json', 'ss': 'tif', 'od': 'json'}
    for task, ext in exts.items():
        if key.endswith(task):
            break

    out_uri = join(collect_dir, key, 'predictions.{}'.format(ext))
    cmd = ['rastervision', 'predict', model_bundle_uri, sample_img, out_uri]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('failure!')
        print(' '.join(cmd))
        exit()


@test.command()
@click.argument('collect_dir')
@click.argument('keys', nargs=-1)
def predict(collect_dir, keys):
    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        if run_all or exp_cfg['key'] in keys:
            key = exp_cfg['key']
            _predict(key, exp_cfg['sample_img'], collect_dir)


if __name__ == '__main__':
    test()
