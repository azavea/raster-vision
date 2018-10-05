from abc import (ABC, abstractmethod)
from typing import (List, Union)
import logging

from google.protobuf import json_format

import rastervision as rv
from rastervision.runner import (CommandDefinition, CommandDAG)
from rastervision.cli import Verbosity

log = logging.getLogger(__name__)


class ExperimentRunner(ABC):
    def log_command(self, command_def, command_id=None, command_dag=None):
        verbosity = Verbosity.get()
        command_type = command_def.command_config.command_type
        experiment_id = command_def.experiment_id
        log.info('{} '.format(command_type))
        log.info('from {}\n'.format(experiment_id))

        if verbosity >= Verbosity.VERBOSE:
            log.info('  INPUTS:')
            for input_uri in command_def.io_def.input_uris:
                log.info('    {}'.format(input_uri))
            if not command_def.io_def.output_uris:
                log.info('  NO OUTPUTS!')
            else:
                log.info('  OUTPUTS:')
                for output_uri in command_def.io_def.output_uris:
                    log.info('    {}'.format(output_uri))
        if verbosity >= Verbosity.VERY_VERBOSE:
            log.debug('  COMMAND CONFIGURATION')
            log.debug('  ---------------------\n')
            log.debug('{}'.format(
                json_format.MessageToJson(
                    command_def.command_config.to_proto())))

        if command_dag:
            upstreams = command_dag.get_upstream_command_ids(command_id)
            if upstreams:
                for upstream_id in upstreams:
                    cdef = command_dag.get_command_definition(upstream_id)
                    msg = '  DEPENDS ON: {} from {}'.format(
                        cdef.command_config.command_type, cdef.experiment_id)
                    log.info(msg)

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

        # Log unrequested commands
        if dry_run:
            if not_requested:
                log.warning('Commands not requsted:')
                for command in not_requested:
                    self.log_command(command)

        # Filter  out commands that don't have any output.
        (command_definitions,
         no_output) = CommandDefinition.filter_no_output(command_definitions)

        # Log commands that have no output
        if dry_run:
            if no_output:
                log.info('Commands not run because they have no output:')
                for command in no_output:
                    self.log_command(command)

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
                msg = ('Commands determined to be '
                       'duplicates based on input and output:')
                log.warning(msg)
                for command in skipped_duplicate_commands:
                    self.log_command(command)

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

        # Log conflicating or alread fulfilled commands
        if dry_run:
            skipped_commands = command_dag.skipped_commands
            if skipped_commands:
                msg = 'Commands skipped because output already exists:'
                log.info(msg)
                for command in skipped_commands:
                    self.log_command(command)

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
            sorted_command_ids = command_dag.get_sorted_command_ids()
            if not any(sorted_command_ids):
                log.error('No commands to run!')
            else:
                log.info('Commands to be run in this order:')
                for command_id in command_dag.get_sorted_command_ids():
                    command_def = command_dag.get_command_definition(
                        command_id)
                    self.log_command(command_def, command_id, command_dag)
        else:
            self._run_experiment(command_dag)

    @abstractmethod
    def _run_experiment(self, command_dag):
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
