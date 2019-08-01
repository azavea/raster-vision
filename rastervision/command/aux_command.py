from rastervision.command import Command


class AuxCommandOptions:

    def __init__(self,
                 split_on=None,
                 inputs=lambda config: None,
                 outputs=lambda config: None,
                 include_by_default=False,
                 required_fields=None):
        """Instantiate an AuxCommandOptions object.

        Args:
            split_on (str): The property of the configuration to use when splitting.
            The configuration at this property must be a list.

            inputs: A function that, given the configuration, returns a list of
            URIs that are inputs into the command. Along with outputs, this allows
            Raster Vision to correctly determine if there are any missing inputs, or
            if the command has already been run. It will also allow the command to
            be run in the right sequence if run with other commands that will produce
            this command's inputs as their outputs.

            outputs: A function that, given the configuration, returns a list of
            URIs that are outputs of the command. See the details on inputs.

            include_by_default: Set this to True if you want this command to run
            by default, meaning it will run every time no specific commands are issued
            on the command line (e.g. how a standard command would run).

            required_fields: Set this to properties of the configuration that are required.
            If the user of the command does not set values into those configuration properties,
            an error will be thrown at configuration building time.

        """
        self.split_on = split_on
        self.inputs = inputs
        self.outputs = outputs
        self.include_by_default = include_by_default
        self.required_fields = required_fields


class AuxCommand(Command):
    """An abstract class representing an auxiliary command.
    The purpose of Aux commands is to make it easy to add
    custom functionality to Raster Vision.
    """

    command_type = None
    options = None

    def __init__(self, command_config):
        self.command_config = command_config
