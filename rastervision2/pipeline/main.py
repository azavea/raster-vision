"""Raster Vision main program"""
import sys
import os
from os.path import join
import logging
import importlib

import click

from rastervision2.pipeline import (_registry, _rv_config, Verbosity)
from rastervision2.pipeline.filesystem import (file_to_json, json_to_file)
from rastervision2.pipeline.config import build_config

log = logging.getLogger(__name__)


def print_error(msg):
    click.echo(click.style(msg, fg='red'), err=True)


def convert_bool_args(args):
    new_args = {}
    for k, v in args.items():
        if v.lower() == 'true':
            v = True
        elif v.lower() == 'false':
            v = False
        else:
            raise ValueError('value of {} must be True or False'.format(k))
        new_args[k] = v
    return new_args


def get_configs(cfg_module, runner, args):
    _get_config = getattr(cfg_module, 'get_config', None)
    _get_configs = _get_config
    if _get_config is None:
        _get_configs = getattr(cfg_module, 'get_configs', None)
    cfgs = _get_configs(runner, **args)
    if not isinstance(cfgs, list):
        cfgs = [cfgs]
    return cfgs


@click.group()
@click.pass_context
@click.option(
    '--profile', '-p', help='Sets the configuration profile name to use.')
@click.option(
    '-v', '--verbose', help='Sets the output to  be verbose.', count=True)
def main(ctx, profile, verbose):
    # Make sure current directory is on PYTHON_PATH
    # so that we can run against modules in current dir.
    sys.path.append(os.curdir)

    # Initialize configuration
    _rv_config.reset(profile=profile, verbosity=verbose + 1)


@main.command('run', short_help='Run sequence of commands within pipeline(s).')
@click.argument('runner')
@click.argument('cfg_path')
@click.argument('commands', nargs=-1)
@click.option(
    '--arg', '-a', type=(str, str), multiple=True, metavar='KEY VALUE')
@click.option('--splits', '-s', default=1)
def run(runner, cfg_path, commands, arg, splits):
    """Run commands within pipelines using runner named RUNNER."""
    tmp_dir_obj = _rv_config.get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    cfg_module = importlib.import_module(cfg_path)
    args = dict(arg)
    args = convert_bool_args(args)
    cfgs = get_configs(cfg_module, runner, args)

    for cfg in cfgs:
        cfg.update()
        # TODO move to update?
        cfg.rv_config = _rv_config.get_config_dict(_registry.rv_config_schema)
        cfg_dict = cfg.dict()
        cfg_json_uri = cfg.get_config_uri()
        json_to_file(cfg_dict, cfg_json_uri)

        pipeline = cfg.build(tmp_dir)
        if not commands:
            commands = pipeline.commands

        runner = _registry.get_runner(runner)()
        runner.run(cfg_json_uri, pipeline, commands, num_splits=splits)


def _run_command(cfg_json_uri, command, split_ind, num_splits, profile=None,
                 verbose=Verbosity.NORMAL):
    tmp_dir_obj = _rv_config.get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    pipeline_cfg_dict = file_to_json(cfg_json_uri)
    rv_config_dict = pipeline_cfg_dict.get('rv_config')
    _rv_config.reset(
        config_overrides=rv_config_dict, profile=profile, verbosity=verbose)

    cfg = build_config(pipeline_cfg_dict)
    pipeline = cfg.build(tmp_dir)

    # TODO generalize this to work outside batch
    if split_ind is None:
        split_ind = int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))
    command_fn = getattr(pipeline, command)

    if num_splits > 1:
        msg = 'Running {} command split {}/{}...'.format(
            command, split_ind + 1, num_splits)
        click.echo(click.style(msg, fg='green'))
        command_fn(split_ind=split_ind, num_splits=num_splits)
    else:
        msg = 'Running {} command...'.format(command)
        click.echo(click.style(msg, fg='green'))
        command_fn()


@main.command(
    'run_command', short_help='Run an individual command within a pipeline.')
@click.pass_context
@click.argument('cfg_json_uri')
@click.argument('command')
@click.option('--split-ind')
@click.option('--num-splits', default=1)
def run_command(ctx, cfg_json_uri, command, split_ind, num_splits):
    profile = ctx.parent.params.get('profile')
    verbose = ctx.parent.params.get('verbose')
    _run_command(cfg_json_uri, command, split_ind, num_splits,
                 profile=profile, verbose=verbose)


if __name__ == '__main__':
    main()
