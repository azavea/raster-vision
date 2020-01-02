"""Raster Vision main program"""
import sys
import os
from os.path import join
import tempfile
import logging
import importlib

import click

from rastervision.v2.core import _registry, _rv_config
from rastervision.v2.core.filesystem import (
    file_to_json, make_dir, json_to_file)
from rastervision.v2.core.config import build_config

log = logging.getLogger(__name__)


def print_error(msg):
    click.echo(click.style(msg, fg='red'), err=True)


@click.group()
@click.option(
    '--profile', '-p', help='Sets the configuration profile name to use.')
@click.option(
    '-v', '--verbose', help='Sets the output to  be verbose.', count=True)
def main(profile, verbose):
    # Make sure current directory is on PYTHON_PATH
    # so that we can run against modules in current dir.
    sys.path.append(os.curdir)

    # Initialize configuration
    _rv_config.update(profile=profile, verbosity=verbose + 1)


@main.command(
    'run', short_help='Run sequence of commands within pipeline(s).')
@click.argument('runner')
@click.argument('cfg_path')
@click.argument('commands', nargs=-1)
@click.option(
    '--arg', '-a', type=(str, str), multiple=True, metavar='KEY VALUE')
@click.option('--splits', '-s', default=1)
def run(runner, cfg_path, commands, arg, splits):
    """Run commands within pipelines using runner named RUNNER."""
    cfg_module = importlib.import_module(cfg_path)
    args = dict(arg)

    get_config = getattr(cfg_module, 'get_config', None)
    get_configs = get_config
    if get_config is None:
        get_configs = getattr(cfg_module, 'get_configs', None)

    new_args = {}
    for k, v in args.items():
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        new_args[k] = v
    args = new_args

    cfgs = get_configs(runner, **args)
    if not isinstance(cfgs, list):
        cfgs = [cfgs]

    for cfg in cfgs:
        cfg.update_all()
        cfg_dict = cfg.dict()
        cfg_json_uri = join(cfg.root_uri, 'pipeline.json')
        json_to_file(cfg_dict, cfg_json_uri)

        pipeline = cfg.get_pipeline()
        if not commands:
            commands = pipeline.commands

        runner = _registry.get_runner(runner)()
        runner.run(cfg_json_uri, pipeline, commands, num_splits=splits)


def _run_command(cfg_json_uri, command, split_ind, num_splits):
    tmp_dir = _rv_config.get_tmp_dir()
    pipeline_cfg_dict = file_to_json(cfg_json_uri)
    cfg = build_config(pipeline_cfg_dict)
    pipeline = cfg.get_pipeline()(cfg, tmp_dir)

    # TODO generalize this to work outside batch
    if split_ind is None:
        split_ind = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
    command_fn = getattr(pipeline, command)

    if num_splits > 1:
        print('Running {} command split {}/{}...'.format(
            command, split_ind + 1, num_splits))
        command_fn(split_ind=split_ind, num_splits=num_splits)
    else:
        print('Running {} command...'.format(command))
        command_fn()


@main.command(
    'run_command', short_help='Run an individual command within a pipeline.')
@click.argument('cfg_json_uri')
@click.argument('command')
@click.option('--split-ind')
@click.option('--num-splits', default=1)
def run_command(cfg_json_uri, command, split_ind, num_splits):
    _run_command(cfg_json_uri, command, split_ind, num_splits)


if __name__ == '__main__':
    main()
