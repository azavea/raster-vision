from rastervision.command import Command


class AuxCommandOptions:
    def __init__(self,
                 split_on=None,
                 inputs=lambda config: None,
                 outputs=lambda config: None,
                 include_by_default=False,
                 required_fields=None):
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
