from abc import (ABC, abstractmethod)


class Evaluator(ABC):
    """Evaluates predictions for a set of scenes.
    """

    @abstractmethod
    def process(self, scenes, tmp_dir):
        pass
