from abc import (ABC, abstractmethod)


class Analyzer(ABC):
    """Class that processes scenes to produce
       some output during the ANALYZE command, which
       can be used for later procesess.

    """

    @abstractmethod
    def process(self, scenes, tmp_dir):
        pass
