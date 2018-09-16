"""Raster Vision main program"""
import sys

import click

import rastervision as rv
from rastervision.experiment import (ExperimentLoader, LoaderError)


def print_error(msg):
    click.echo(click.style(msg, fg='red'), err=True)


@click.group()
@click.option(
    '--profile', '-p', help='Sets the configuration profile name to use.')
@click.option('--verbose', '-v', is_flag=True)
def main(profile, verbose):
    # TODO: implement verbose
    if profile:
        rv._registry.initialize_config(profile=profile)


@main.command()
@click.argument('runner')
@click.argument('commands', nargs=-1)
@click.option(
    '--experiment_module',
    '-e',
    help=('Name of an importable module to look for experiment sets '
          'in. If not supplied, experiments will be loaded '
          'from __main__'))
@click.option(
    '--dry-run',
    '-n',
    is_flag=True,
    help=('Execute a dry run, which will print out information '
          'about the commands to be run, but will not actually '
          'run the commands'))
@click.option(
    '--arg',
    '-a',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    help=('Pass a parameter to the experiments if the method '
          'parameter list takes in a parameter with that key. '
          'Multiple args can be supplied'))
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
@click.option('--rv-branch')
def run(runner, commands, experiment_module, dry_run, arg, rerun, rv_branch):
    # Validate runner
    valid_runners = list(
        map(lambda x: x.lower(), rv.ExperimentRunner.list_runners()))
    if runner not in valid_runners:
        print_error('Invalid experiment runner: "{}". '
                    'Must be one of: "{}"'.format(runner,
                                                  '", "'.join(valid_runners)))
        sys.exit(1)

    runner = rv.ExperimentRunner.get_runner(runner)

    if experiment_module:
        module_to_load = experiment_module
    else:
        module_to_load = '__main__'

    if not commands:
        commands = rv.ALL_COMMANDS
    else:
        commands = list(map(lambda x: x.upper(), commands))

    experiment_args = {}
    for k, v in arg:
        experiment_args[k] = v

    loader = ExperimentLoader(experiment_args=experiment_args)
    try:
        experiments = loader.load_from_module(module_to_load)
    except LoaderError as e:
        print_error(str(e))
        sys.exit(1)

    if not experiments:
        if experiment_module:
            print_error(
                'No experiments found in {}.'.format(experiment_module))
        else:
            print_error('No experiments found.')

    runner.run(experiments, commands_to_run=commands, rerun_commands=rerun)


@main.command()
@click.option(
    '--experiment_module',
    '-e',
    help=('Name of an importable module to look for experiment sets '
          'in. If not supplied, experiments will be loaded '
          'from __main__'))
@click.option(
    '--arg',
    '-a',
    type=(str, str),
    multiple=True,
    metavar='KEY VALUE',
    help=('Pass a parameter to the experiments if the method '
          'parameter list takes in a parameter with that key. '
          'Multiple args can be supplied'))
def ls(experiment_module, arg):
    if experiment_module:
        module_to_load = experiment_module
    else:
        module_to_load = '__main__'

    experiment_args = {}
    for k, v in arg:
        experiment_args[k] = v

    loader = ExperimentLoader(experiment_args=experiment_args)
    try:
        experiments = loader.load_from_module(module_to_load)
    except LoaderError as e:
        print_error(str(e))
        sys.exit(1)

    if not experiments:
        if experiment_module:
            print_error(
                'No experiments found in {}.'.format(experiment_module))
        else:
            print_error('No experiments found.')

    for e in experiments:
        click.echo('{}'.format(e.id))


if __name__ == '__main__':
    main()
