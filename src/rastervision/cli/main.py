"""Raster Vision main program"""
import sys
from tempfile import TemporaryDirectory

import click

import rastervision as rv
from rastervision.experiment import (ExperimentLoader, LoaderError)
from rastervision.runner import (ExperimentRunner)


def print_error(msg):
    click.echo(click.style(msg, fg='red'), err=True)


@click.group()
@click.option(
    '--profile', '-p', help='Sets the configuration profile name to use.')
def main(profile):
    # Initialize configuration
    if profile:
        rv._registry.initialize_config(profile=profile)


@main.command(
    'run', short_help='Run Raster Vision commands against Experiments.')
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
    '--skip-file-check',
    '-x',
    is_flag=True,
    help=('Skip the step that verifies that file exist.'))
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
    '--prefix',
    metavar='PREFIX',
    default="exp_",
    help=('Prefix for methods containing experiments. (default: "exp_")'))
@click.option(
    '--method',
    '-m',
    'methods',
    multiple=True,
    metavar='PATTERN',
    help=('Pattern to match method names to run.'))
@click.option(
    '--filter',
    '-f',
    'filters',
    multiple=True,
    metavar='PATTERN',
    help=('Pattern to match experiment names to run.'))
@click.option(
    '--rerun',
    '-r',
    is_flag=True,
    default=False,
    help=('Rerun commands, regardless if '
          'their output files already exist.'))
def run(runner, commands, experiment_module, dry_run, skip_file_check, arg,
        prefix, methods, filters, rerun):
    """Run Raster Vision commands from experiments, using the
    experiment runner named RUNNER."""
    # Validate runner
    valid_runners = list(
        map(lambda x: x.lower(), rv.ExperimentRunner.list_runners()))
    if runner not in valid_runners:
        print_error('Invalid experiment runner: "{}". '
                    'Must be one of: "{}"'.format(runner,
                                                  '", "'.join(valid_runners)))
        sys.exit(1)

    runner = ExperimentRunner.get_runner(runner)

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

    loader = ExperimentLoader(experiment_args=experiment_args,
                              experiment_method_prefix=prefix,
                              experiment_method_patterns=methods,
                              experiment_name_patterns=filters)
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

    runner.run(
        experiments,
        commands_to_run=commands,
        rerun_commands=rerun,
        skip_file_check=skip_file_check)


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
    """Print out a list of Experiment IDs."""
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


@main.command(
    'predict', short_help='Make predictions using a predict package.')
@click.argument('predict_package', type=click.Path(exists=True))
@click.argument('image_uri', type=click.Path(exists=True))
@click.argument('output_uri', type=click.Path(exists=False))
@click.option(
    '--update_stats',
    '-a',
    is_flag=True,
    help=('Run an analysis on this individual image, as '
          'opposed to using any analysis like statistics '
          'that exist in the prediction package'))
@click.option(
    '--channel-order',
    help='String containing channel_order.' + ' Example: \"2 1 0\"')
def predict(predict_package, image_uri, output_uri, update_stats,
            channel_order):
    """Make predictions on the image at IMAGE_URI
    using PREDICT_PACKAGE and store the
    prediciton output at OUTPUT_URI.
    """
    if channel_order is not None:
        channel_order = [
            int(channel_ind) for channel_ind in channel_order.split(' ')
        ]
    with TemporaryDirectory() as tmp_dir:
        predict = rv.Predictor(predict_package, tmp_dir, update_stats,
                               channel_order).predict
        predict(image_uri, output_uri)


@main.command(
    'run_command', short_help='Run a command from configuration file.')
@click.argument('command_config_uri')
def run_command(command_config_uri):
    """Run a command from a serialized command configuration
    at COMMAND_CONFIG_URI.
    """
    rv.runner.CommandRunner.run(command_config_uri)


if __name__ == '__main__':
    main()
