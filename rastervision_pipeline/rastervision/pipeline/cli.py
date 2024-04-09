import sys
import os
import logging
import importlib
import importlib.util
from typing import List, Dict, Optional, Tuple

import click

from rastervision.pipeline import (registry_ as registry, rv_config_ as
                                   rv_config)
from rastervision.pipeline.file_system import (file_to_json, get_tmp_dir)
from rastervision.pipeline.config import build_config, save_pipeline_config
from rastervision.pipeline.pipeline_config import PipelineConfig

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
        new_args[k] = v
    return new_args


def get_configs(cfg_module_path: str, runner: str,
                args: Dict[str, any]) -> List[PipelineConfig]:
    """Get PipelineConfigs from a module.

    Calls a get_config(s) function with some arguments from the CLI
    to get a list of PipelineConfigs.

    Args:
        cfg_module_path: the module with `get_configs` function that returns
            PipelineConfigs. This can either be a Python module path or a local path to
            a .py file.
        runner: name of the runner
        args: CLI args to pass to the get_config(s) function that comes from
            the --args option
    """
    if cfg_module_path.endswith('.py'):
        # From https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # noqa
        spec = importlib.util.spec_from_file_location('rastervision.pipeline',
                                                      cfg_module_path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
    else:
        cfg_module = importlib.import_module(cfg_module_path)

    _get_config = getattr(cfg_module, 'get_config', None)
    _get_configs = _get_config
    if _get_config is None:
        _get_configs = getattr(cfg_module, 'get_configs', None)
    if _get_configs is None:
        raise Exception('There must be a get_config or get_configs function '
                        f'in {cfg_module_path}.')
    cfgs = _get_configs(runner, **args)
    if not isinstance(cfgs, list):
        cfgs = [cfgs]

    for cfg in cfgs:
        if not issubclass(type(cfg), PipelineConfig):
            raise Exception('All objects returned by get_configs in '
                            f'{cfg_module_path} must be PipelineConfigs.')
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


def _run_pipeline(cfg,
                  runner,
                  tmp_dir,
                  splits=1,
                  commands=None,
                  pipeline_run_name: str = 'raster-vision'):
    cfg.update()
    cfg.recursive_validate_config()
    # This is to run the validation again to check any fields that may have changed
    # after the Config was constructed, possibly by the update method.
    build_config(cfg.dict())
    cfg_json_uri = cfg.get_config_uri()
    save_pipeline_config(cfg, cfg_json_uri)
    pipeline = cfg.build(tmp_dir)
    if not commands:
        commands = pipeline.commands

    click.secho(f'Stages to run: {commands}', fg='white', bold=True)

    runner.run(
        cfg_json_uri,
        pipeline,
        commands,
        num_splits=splits,
        pipeline_run_name=pipeline_run_name)


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
@click.option(
    '--pipeline-run-name',
    default='raster-vision',
    help='The name for this run of the pipeline.')
def run(runner: str, cfg_module: str, commands: List[str],
        arg: List[Tuple[str, str]], splits: int, pipeline_run_name: str):
    """Run COMMANDS within pipelines in CFG_MODULE using RUNNER.

    RUNNER: name of the Runner to use

    CFG_MODULE: the module with `get_configs` function that returns PipelineConfigs.
    This can either be a Python module path or a local path to a .py file.

    COMMANDS: space separated sequence of commands to run within pipeline. The order in
    which to run them is based on the Pipeline.commands attribute. If this is omitted,
    all commands will be run.
    """
    tmp_dir_obj = get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    args = dict(arg)
    args = convert_bool_args(args)
    cfgs = get_configs(cfg_module, runner, args)
    runner = registry.get_runner(runner)()

    for cfg in cfgs:
        _run_pipeline(cfg, runner, tmp_dir, splits, commands,
                      pipeline_run_name)


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

    tmp_dir_obj = get_tmp_dir()
    tmp_dir = tmp_dir_obj.name

    cfg = build_config(pipeline_cfg_dict)
    pipeline = cfg.build(tmp_dir)

    if num_splits is not None and split_ind is None and runner is not None:
        runner = registry.get_runner(runner)()
        split_ind = runner.get_split_ind()

    command_fn = getattr(pipeline, command)

    if num_splits is not None and num_splits > 1:
        msg = f'Running {command} command split {split_ind + 1}/{num_splits}...'
        click.secho(msg, fg='green', bold=True)
        command_fn(split_ind=split_ind, num_splits=num_splits)
    else:
        msg = f'Running {command} command...'
        click.secho(msg, fg='green', bold=True)
        command_fn()


@main.command(
    'run_command', short_help='Run an individual command within a pipeline.')
@click.argument('cfg_json_uri')
@click.argument('command')
@click.option(
    '--split-ind', type=int, help='The process index of a split command')
@click.option(
    '--num-splits',
    type=int,
    help='The number of processes to use for running splittable commands')
@click.option(
    '--runner', type=str, help='Name of runner to use', default='inprocess')
def run_command(cfg_json_uri: str, command: str, split_ind: Optional[int],
                num_splits: Optional[int], runner: str):
    """Run a single COMMAND using a serialized PipelineConfig in CFG_JSON_URI."""
    _run_command(
        cfg_json_uri,
        command,
        split_ind=split_ind,
        num_splits=num_splits,
        runner=runner)


def _main():  # pragma: no cover
    for pc in registry.get_plugin_commands():
        main.add_command(pc)
    main()


if __name__ == '__main__':
    _main()
