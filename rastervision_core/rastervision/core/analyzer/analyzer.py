from typing import List
from abc import (ABC, abstractmethod)

from rastervision.core.data import Scene


class Analyzer(ABC):
    """Analyzes scenes and writes some output while running the analyze command.

    This output can be used to normalize images, for example.
    """

    @abstractmethod
    def process(self, scenes: List[Scene], tmp_dir: str):
        """Process scenes and save result."""
