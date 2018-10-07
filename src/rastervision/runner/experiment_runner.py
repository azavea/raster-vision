from abc import (ABC, abstractmethod)
from typing import (List, Union)

import click
from google.protobuf import json_format

import rastervision as rv
from rastervision.runner import (CommandDefinition, CommandDAG)
from rastervision.cli import Verbosity


class ExperimentRunner(ABC):
    def print_command(self, command_def, command_id=None, command_dag=None):
        verbosity = Verbosity.get()
        command_type = command_def.command_config.command_type
        experiment_id = command_def.experiment_id
        click.echo(
            click.style('{} '.format(command_type), bold=True), nl=False)
        click.echo('from {}'.format(experiment_id))

        if verbosity >= Verbosity.VERBOSE:
            click.echo('  INPUTS:')
            for input_uri in command_def.io_def.input_uris:
                click.echo('    {}'.format(input_uri))
            if not command_def.io_def.output_uris:
                click.echo('  NO OUTPUTS!')
            else:
                click.echo('  OUTPUTS:')
                for output_uri in command_def.io_def.output_uris:
                    click.echo('    {}'.format(output_uri))
        if verbosity >= Verbosity.VERY_VERBOSE:
            click.echo(click.style('  COMMAND CONFIGURATION', bold=True))
            click.echo('  ---------------------\n')
            click.echo('{}'.format(
                json_format.MessageToJson(
                    command_def.command_config.to_proto())))

        if command_dag:
            upstreams = command_dag.get_upstream_command_ids(command_id)
            if upstreams:
                for upstream_id in upstreams:
                    cdef = command_dag.get_command_definition(upstream_id)
                    msg = '  DEPENDS ON: {} from {}'.format(
                        cdef.command_config.command_type, cdef.experiment_id)
                    click.echo(click.style(msg, fg='cyan'))

    def run(self,
            experiments: Union[List[rv.ExperimentConfig], rv.ExperimentConfig],
            commands_to_run=rv.ALL_COMMANDS,
            rerun_commands=False,
            skip_file_check=False,
            dry_run: bool = False):
        if not isinstance(experiments, list):
            experiments = [experiments]

        command_definitions = CommandDefinition.from_experiments(experiments)

        # Filter  out commands we aren't running.
        command_definitions, not_requested = CommandDefinition.filter_to_target_commands(
            command_definitions, commands_to_run)

        # Print unrequested commands
        if dry_run:
            if not_requested:
                print()
                click.echo(
                    click.style(
                        'Commands not requsted:', fg='yellow', underline=True))
                for command in not_requested:
                    self.print_command(command)
                    print()

        # Filter  out commands that don't have any output.
        (command_definitions,
         no_output) = CommandDefinition.filter_no_output(command_definitions)

        # Print commands that have no output
        if dry_run:
            if no_output:
                print()
                click.echo(
                    click.style(
                        'Commands not run because they have no output:',
                        fg='yellow',
                        bold=True,
                        underline=True))
                for command in no_output:
                    self.print_command(command)
                print()

        # Check if there are any unsatisfied inputs.
        missing_inputs = CommandDefinition.get_missing_inputs(
            command_definitions)
        if missing_inputs:
            # TODO: Replace with logging?
            s = ''
            for exp_id in missing_inputs:
                s += 'In {}:\n\t{}\n'.format(
                    exp_id, '\t{}\n'.join(missing_inputs[exp_id]))

            raise rv.ConfigError('There were missing input URIs '
                                 'that are required, but were not '
                                 'able to be derived: \n{}'.format(s))

        # Remove duplicate commands, defining equality for a command by
        # the tuple (command_type, input_uris, output_uris)
        (unique_commands, skipped_duplicate_commands
         ) = CommandDefinition.remove_duplicates(command_definitions)

        if dry_run:
            if skipped_duplicate_commands:
                print()
                msg = ('Commands determined to be '
                       'duplicates based on input and output:')
                click.echo(
                    click.style(msg, fg='yellow', bold=True, underline=True))
                for command in skipped_duplicate_commands:
                    self.print_command(command)
                print()

        # Ensure that for each type of command, there are none that clobber
        # each other's output.
        clashing_commands = CommandDefinition.get_clashing_commands(
            unique_commands)

        if clashing_commands:
            clashing_msgs = []
            for (output_uri, c_defs) in clashing_commands:
                command_type = c_defs[0].command_config.command_type
                experiments = ', '.join(map(lambda c: c.experiment_id, c_defs))
                clashing_msgs.append(
                    'The {} command in the following experiments '
                    'output {}, but are not equal: {}'.format(
                        command_type, output_uri, experiments))
            # TODO: Replace with logging?
            s = '\t\n'.join(clashing_msgs)

            raise rv.ConfigError('ERROR: Command outputs will'
                                 'override each other: \n{}\n'.format(s))

        command_dag = CommandDAG(
            unique_commands, rerun_commands, skip_file_check=skip_file_check)

        # Print conflicating or alread fulfilled commands
        if dry_run:
            skipped_commands = command_dag.skipped_commands
            if skipped_commands:
                print()
                msg = 'Commands skipped because output already exists:'
                click.echo(
                    click.style(msg, fg='yellow', bold=True, underline=True))
                for command in skipped_commands:
                    self.print_command(command)
                print()

        # Save experiment configs
        experiments_by_id = dict(map(lambda e: (e.id, e), experiments))
        seen_ids = set([])
        for command_def in command_dag.get_command_definitions():
            if command_def.experiment_id not in seen_ids:
                seen_ids.add(command_def.experiment_id)
                experiment = experiments_by_id[command_def.experiment_id]
                if not dry_run:
                    experiment.fully_resolve().save_config()

        if dry_run:
            print()
            sorted_command_ids = command_dag.get_sorted_command_ids()
            if len(sorted_command_ids) == 0:
                click.echo(
                    click.style('No commands to run!', fg='red', bold=True))
                print()
            else:
                click.echo(
                    click.style(
                        'Commands to be run in this order:',
                        fg='green',
                        bold=True,
                        underline=True))
                for command_id in command_dag.get_sorted_command_ids():
                    command_def = command_dag.get_command_definition(
                        command_id)
                    self.print_command(command_def, command_id, command_dag)
                    print()
            self._dry_run(command_dag)
        else:
            self._run_experiment(command_dag)

    @abstractmethod
    def _run_experiment(self, command_dag):
        pass

    def _dry_run(self, command_dag):
        """Overridden by subclasses if they contribute dry run information.
        """
        pass

    @staticmethod
    def get_runner(runner_type):
        """Gets the runner associated with this runner type."""
        # Runner keys are upper cased.
        return rv._registry.get_experiment_runner(runner_type.upper())

    @staticmethod
    def list_runners():
        """Returns a list of valid runner keys."""
        return rv._registry.get_experiment_runner_keys()
