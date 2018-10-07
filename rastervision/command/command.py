from abc import ABC, abstractmethod

from rastervision.rv_config import RVConfig


class Command(ABC):
    @abstractmethod
    def run(self, tmp_dir):
        """Run the command."""
        pass

    def set_tmp_dir(self, tmp_dir):
        self._tmp_dir = tmp_dir

    def get_tmp_dir(self):
        if hasattr(self, '_tmp_dir') and self._tmp_dir:
            if isinstance(self._tmp_dir, str):
                return self._tmp_dir
            else:
                return self._tmp_dir.name
        else:
            return RVConfig.get_tmp_dir().name


class NoOpCommand(Command):
    """Defines a command that does nothing.
    """

    def run(self, tmp_dir):
        pass
