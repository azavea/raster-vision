import sys
import os
import logging
import importlib
from typing import List, Dict, Optional, Tuple
from types import ModuleType

import click

from rastervision2.pipeline import (registry, rv_config)
from rastervision2.pipeline.file_system import (file_to_json, json_to_file)
from rastervision2.pipeline.config import build_config

log = logging.getLogger(__name__)


def print_error(msg):
    """Print error message to console in red."""
    click.echo(click.style(msg, fg='red'), err=True)


def convert_bool_args(args: dict) -> dict:
    """Convert boolean CLI arguments from string to bool.

    Args:
        args: a mapping from CLI argument names to values

    Returns:
        copy of args with boolean string values convert to bool
    """
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


def get_configs(cfg_module: ModuleType, runner: str, args: Dict[str, any]
                ) -> List['rastervision2.pipeline.PipelineConfig']:  # noqa
    """Get PipelineConfigs from a module.

    Calls a get_config(s) function with some arguments from the CLI
    to get a list of PipelineConfigs.

    Args:
        cfg_module: a module with a get_config(s) function
        runner: name of the runner
        args: CLI args to pass to the get_config(s) function that comes from
            the --args option
    """
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
    '-v', '--verbose', help='Increment the verbosity level.', count=True)
@click.option('--tmpdir', help='Root of temporary directories to use.')
def main(ctx: click.Context, profile: Optional[str], verbose: int,
         tmpdir: str):
    """The main click command.

    Sets the profile, verbosity, and tmp_dir in RVConfig.
    """
    # Make sure current directory is on PYTHON_PATH
    # so that we can run against modules in current dir.
    sys.path.append(os.curdir)
    rv_config.set_verbosity(verbosity=verbose + 1)
    rv_config.set_tmp_dir_root(tmp_dir_root=tmpdir)
    rv_config.set_everett_config(profile=profile)


@main.command('run', short_help='Run sequence of commands within pipeline(s).')
@click.argument('runner')
@click.argument('cfg_module')
@click.argument('commands', nargs=-1)
@click.option(
    '--arg',
    '-a',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    help='Arguments to pass to get_config function')
@click.option(
    '--splits',
    '-s',
    default=1,
    help='Number of splits to run in parallel for splittable commands')
def run(runner: str, cfg_module: str, commands: List[str],
        arg: List[Tuple[str, str]], splits: int):
    """Subcommand to run commands within pipelines using runner named RUNNER.

    Args:
        runner: name of runner to use
        cfg_module: name of module with `get_configs` function that returns
            PipelineConfigs
        commands: names of commands to run within pipeline. The order in which
            to run them is based on the Pipeline.commands attribute. If this is
            omitted, all commands will be run.
    """
    tmp_dir_obj = rv_config.get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    cfg_module = importlib.import_module(cfg_module)
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


def _run_command(cfg_json_uri: str,
                 command: str,
                 split_ind: Optional[int] = None,
                 num_splits: Optional[int] = None,
                 runner: Optional[str] = None):
    """Run a single command using a serialized PipelineConfig.

    Args:
        cfg_json_uri: URI of a JSON file with a serialized PipelineConfig
        command: name of command to run
        split_ind: the index that a split command should assume
        num_splits: the total number of splits to use
        runner: the name of the runner to use
    """
    pipeline_cfg_dict = file_to_json(cfg_json_uri)
    rv_config_dict = pipeline_cfg_dict.get('rv_config')
    rv_config.set_everett_config(
        profile=rv_config.profile, config_overrides=rv_config_dict)

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
@click.argument('cfg_json_uri')
@click.argument('command')
@click.option('--split-ind', type=int, help='The index of a split command')
@click.option(
    '--num-splits',
    type=int,
    help='The number of splits to use if command is split')
@click.option('--runner', type=str, help='Name of runner to use')
def run_command(cfg_json_uri: str, command: str, split_ind: Optional[int],
                num_splits: Optional[int], runner: str):
    """Run a single command using a serialized PipelineConfig.

    Args:
        cfg_json_uri: URI of a JSON file with a serialized PipelineConfig
        command: name of command to run
    """
    _run_command(
        cfg_json_uri,
        command,
        split_ind=split_ind,
        num_splits=num_splits,
        runner=runner)


if __name__ == '__main__':
    main()
