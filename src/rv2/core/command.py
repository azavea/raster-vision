from abc import ABC, abstractmethod


class Command(ABC):
    """A command which can be run from the command line."""

    @abstractmethod
    def run(self):
        """Run the command."""
        pass
