import rastervision as rv

from abc import (ABC, abstractmethod)
from typing import (List, Union)

from rastervision.runner import (CommandDefinition, CommandDAG)


class ExperimentRunner(ABC):
    def run(self,
            experiments: Union[List[rv.ExperimentConfig], rv.ExperimentConfig],
            commands_to_run=rv.ALL_COMMANDS,
            rerun_commands=False,
            skip_file_check=False,
            dry_run: bool = False):
        if not isinstance(experiments, list):
            experiments = [experiments]

        _command_definitions = CommandDefinition.from_experiments(experiments)

        # Filter  out commands we aren't running.
        command_definitions = CommandDefinition.filter_commands(
            _command_definitions, commands_to_run)

        # Print unrequested commands
        if dry_run:
            not_requested = set(_command_definitions) - set(command_definitions)
            if not_requested:
                print()
                print('Not requsted:')
                for command in not_requested:
                    command_type = command.command_config.command_type
                    experiment_id = command.experiment_id
                    print('{} from {}'.format(command_type, experiment_id))
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
        (unique_commands, skipped_commands
         ) = CommandDefinition.remove_duplicates(command_definitions)

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
                    'The {} command in the follwoing experiments '
                    'output {}, but are not equal: {}'.format(
                        command_type, output_uri, experiments))
            # TODO: Replace with logging?
            s = '\t\n'.join(clashing_msgs)

            raise rv.ConfigurationError(
                'ERROR: Command outputs will'
                'override each other: \n{}\n'.format(s))

        command_dag = CommandDAG(
            unique_commands, rerun_commands, skip_file_check=skip_file_check)

        # Print conflicating or alread fulfilled commands
        if dry_run:
            skipped_commands.extend(command_dag.skipped_commands)
            if skipped_commands:
                print()
                print('Skipped due to output conflicts:')
                for command in skipped_commands:
                    command_type = command.command_config.command_type
                    experiment_id = command.experiment_id
                    print('{} from {}'.format(command_type, experiment_id))
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
            print('To be run:')
            for command in command_dag.get_sorted_commands():
                command_type = command.command_type
                root_uri = command.root_uri
                print('{} in {}'.format(command_type, root_uri))
            print()
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
