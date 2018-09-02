from abc import ABC, abstractmethod


class Command(ABC):
    """A Raster Vision command, which transforms inputs
       and creates outputs.
    """

    @abstractmethod
    def get_inputs(self):
        """Get the inputs for this command.

        Returns: A list of URIs that are the inputs
                 to this command.
        """
        pass

    @abstractmethod
    def get_ouputs(self):
        """Get the inputs for this command.

        Returns: A list of  URIs that are the outputs
                 of this  command.
        """
        pass

    @abstractmethod
    def run(self):
        """Run the command."""
        pass
