from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def run(self, tmp_dir):
        """Run the command."""
        pass

    def set_tmp_dir(self, tmp_dir):
        self._tmp_dir = tmp_dir

    def get_tmp_dir(self):
        return self._tmp_dir


class NoOpCommand(Command):
    """Defines a command that does nothing.
    """

    def run(self, tmp_dir):
        pass
