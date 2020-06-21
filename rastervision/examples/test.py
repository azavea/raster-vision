# flake8: noqa

import subprocess
from os.path import join, basename, dirname

import click

from rastervision.pipeline.file_system import (
    list_paths, download_or_copy, file_exists, make_dir, file_to_json)

cfg = [
    {
        'key': 'spacenet-rio-cc',
        'module': 'rastervision.examples.chip_classification.spacenet_rio',
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
    },
    {
        'key': 'isprs-potsdam-ss',
        'module': 'rastervision.examples.semantic_segmentation.isprs_potsdam',
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
    },
    {
        'key': 'cowc-potsdam-od',
        'module': 'rastervision.examples.object_detection.cowc_potsdam',
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
    },
    {
        'key': 'spacenet-vegas-buildings-ss',
        'module': 'rastervision.examples.semantic_segmentation.spacenet_vegas',
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
    },
    {
        'key': 'spacenet-vegas-roads-ss',
        'module': 'rastervision.examples.semantic_segmentation.spacenet_vegas',
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
    },
    {
        'key': 'xview-od',
        'module': 'rastervision.examples.object_detection.xview',
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
    },
]


def run_experiment(exp_cfg, output_dir, test=True, remote=False,
                   commands=None):
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
        cmd += ['--splits', '4']

    print('running command:')
    print(' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('failure!')
        print(' '.join(cmd))
        exit()


def collect_experiment(key, root_uri, output_dir, get_pred_package=False):
    print('\nCollecting experiment {}...\n'.format(key))

    if root_uri.startswith('s3://'):
        predict_package_uris = list_paths(
            join(root_uri, key, 'bundle'), ext='predict_package.zip')
        eval_json_uris = list_paths(
            join(root_uri, key, 'eval'), ext='eval.json')
    else:
        predict_package_uris = glob.glob(
            join(root_uri, key, 'bundle', '*', 'predict_package.zip'))
        eval_json_uris = glob.glob(
            join(root_uri, key, 'eval', '*', 'eval.json'))

    if len(predict_package_uris) > 1 or len(eval_json_uris) > 1:
        print('Cannot collect from key with multiple experiments!!!')
        return

    if len(predict_package_uris) == 0 or len(eval_json_uris) == 0:
        print('Missing output!!!')
        return

    predict_package_uri = predict_package_uris[0]
    eval_json_uri = eval_json_uris[0]
    make_dir(join(output_dir, key))
    if get_pred_package:
        download_or_copy(predict_package_uri, join(output_dir, key))

    download_or_copy(eval_json_uri, join(output_dir, key))

    eval_json = file_to_json(join(output_dir, key, 'eval.json'))
    pprint.pprint(eval_json['overall'], indent=4)


def validate_keys(keys):
    exp_keys = [exp_cfg['key'] for exp_cfg in cfg]
    invalid_keys = set(keys).difference(exp_keys)
    if invalid_keys:
        raise ValueError('{} are invalid keys'.format(', '.join(invalid_keys)))


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
            run_experiment(
                exp_cfg,
                output_dir,
                test=test,
                remote=remote,
                commands=commands)


@test.command()
@click.argument('root_uri')
@click.argument('output_dir')
@click.argument('keys', nargs=-1)
@click.option('--get-pred-package', is_flag=True)
def collect(root_uri, output_dir, keys, get_pred_package):
    run_all = len(keys) == 0
    validate_keys(keys)

    for exp_cfg in cfg:
        key = exp_cfg['key']
        if run_all or key in keys:
            collect_experiment(key, root_uri, output_dir, get_pred_package)


@test.command()
@click.argument('root_uri')
def collect_eval_dir(root_uri):
    eval_json_uris = list_paths(join(root_uri, 'eval'), ext='eval.json')
    for eval_json_uri in eval_json_uris:
        eval_json = file_to_json(eval_json_uri)
        print(basename(dirname(eval_json_uri)))
        print(eval_json['overall'][-1]['f1'])
        print()


if __name__ == '__main__':
    test()
