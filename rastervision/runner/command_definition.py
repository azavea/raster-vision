from typing import List

import rastervision as rv


class CommandDefinition:
    def __init__(self, experiment_id, command_config, io_def):
        self.experiment_id = experiment_id
        self.command_config = command_config
        self.io_def = io_def

    def _key(self):
        return (self.command_config.command_type, '|'.join(
            sorted(self.io_def.input_uris)), '|'.join(
                sorted(self.io_def.output_uris)))

    def __eq__(self, other):
        return self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    @classmethod
    def from_experiments(cls, experiments: List[rv.ExperimentConfig]):
        command_definitions = []

        for experiment in experiments:
            e = experiment
            for command_type in rv.ALL_COMMANDS:
                (e, io_def) = e.update_for_command(command_type, e)
                command_config = e.make_command_config(command_type)
                command_def = cls(e.id, command_config, io_def)
                command_definitions.append(command_def)

        return command_definitions

    @staticmethod
    def filter_to_target_commands(command_definitions, target_commands):
        """Filters commands by the target command type."""
        result = []
        skipped = []
        for command_def in command_definitions:
            if command_def.command_config.command_type in target_commands:
                result.append(command_def)
            else:
                skipped.append(command_def)

        return (result, skipped)

    @staticmethod
    def filter_no_output(command_definitions):
        """Filters commands that have no output."""
        result = []
        skipped = []
        for command_def in command_definitions:
            if any(command_def.io_def.output_uris):
                result.append(command_def)
            else:
                skipped.append(command_def)

        return (result, skipped)

    @staticmethod
    def remove_duplicates(command_definitions):
        """Remove duplicate commands.

        Removes duplicated commands, defining equality for a command by
        the tuple (command_type, input_uris, output_uris)
        """

        unique_commands = []
        skipped_commands = []
        seen_commands = set([])
        for command_def in command_definitions:
            if command_def not in seen_commands:
                seen_commands.add(command_def)
                unique_commands.append(command_def)
            else:
                skipped_commands.append(command_def)

        return (unique_commands, skipped_commands)

    @staticmethod
    def get_missing_inputs(command_definitions):
        """Gathers missing inputs from a set of commands.

        Returns a dictionary of experiment id to list of missing input URIs.
        """
        missing_inputs = {}
        for command_def in command_definitions:
            if command_def.io_def.missing_input_messages:
                mi = command_def.io_def.missing_input_messages
                missing_inputs[command_def.experiment_id] = mi
        return missing_inputs

    @staticmethod
    def get_clashing_commands(command_definitions):
        """Reports commands that will overwrite each other's outputs.

        Only reports commands as clashing if they are of the same command type.

        Returns a List[str, List[CommandDefinition]] of output URIs
        and clashing commands.
        """
        outputs_to_defs = {}
        clashing_commands = []
        for command_def in command_definitions:
            command_type = command_def.command_config.command_type
            for output_uri in command_def.io_def.output_uris:
                if (output_uri, command_type) not in outputs_to_defs:
                    outputs_to_defs[(output_uri, command_type)] = []
                outputs_to_defs[(output_uri, command_type)].append(command_def)

        for ((output_uri, _), command_defs) in outputs_to_defs.items():
            if len(command_defs) > 1:
                clashing_commands.append((output_uri, command_defs))

        return clashing_commands
