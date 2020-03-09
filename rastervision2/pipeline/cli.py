"""Raster Vision main program"""
import sys
import os
import logging
import importlib

import click

from rastervision2.pipeline import (registry, rv_config)
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
@click.option('--tmpdir', help='Root of temporary directories to use.')
def main(ctx, profile, verbose, tmpdir):
    # Make sure current directory is on PYTHON_PATH
    # so that we can run against modules in current dir.
    sys.path.append(os.curdir)

    # Initialize configuration
    rv_config.reset(profile=profile, verbosity=verbose + 1, tmp_dir=tmpdir)


@main.command('run', short_help='Run sequence of commands within pipeline(s).')
@click.argument('runner')
@click.argument('cfg_path')
@click.argument('commands', nargs=-1)
@click.option(
    '--arg', '-a', type=(str, str), multiple=True, metavar='KEY VALUE')
@click.option('--splits', '-s', default=1)
def run(runner, cfg_path, commands, arg, splits):
    """Run commands within pipelines using runner named RUNNER."""
    tmp_dir_obj = rv_config.get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    cfg_module = importlib.import_module(cfg_path)
    args = dict(arg)
    args = convert_bool_args(args)
    cfgs = get_configs(cfg_module, runner, args)
    runner = registry.get_runner(runner)()

    for cfg in cfgs:
        cfg.update()
        cfg.rv_config = rv_config.get_config_dict(registry.rv_config_schema)
        cfg.recursive_validate_config()

        cfg_dict = cfg.dict()
        cfg_json_uri = cfg.get_config_uri()
        json_to_file(cfg_dict, cfg_json_uri)

        pipeline = cfg.build(tmp_dir)
        if not commands:
            commands = pipeline.commands

        runner.run(cfg_json_uri, pipeline, commands, num_splits=splits)


def _run_command(cfg_json_uri,
                 command,
                 split_ind=None,
                 num_splits=None,
                 runner=None):
    pipeline_cfg_dict = file_to_json(cfg_json_uri)
    rv_config_dict = pipeline_cfg_dict.get('rv_config')
    rv_config.reset(
        config_overrides=rv_config_dict,
        verbosity=rv_config.verbosity,
        tmp_dir=rv_config.tmp_dir)

    tmp_dir_obj = rv_config.get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    cfg = build_config(pipeline_cfg_dict)
    pipeline = cfg.build(tmp_dir)

    if num_splits is not None and split_ind is None and runner is not None:
        runner = registry.get_runner(runner)()
        split_ind = runner.get_split_ind()

    command_fn = getattr(pipeline, command)

    if num_splits is not None and num_splits > 1:
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
@click.option('--split-ind', type=int)
@click.option('--num-splits', type=int)
@click.option('--runner', type=str)
def run_command(ctx, cfg_json_uri, command, split_ind, num_splits, runner):
    _run_command(
        cfg_json_uri,
        command,
        split_ind=split_ind,
        num_splits=num_splits,
        runner=runner)


if __name__ == '__main__':
    main()
