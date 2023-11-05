from abc import (ABC, abstractmethod)
from pprint import pformat


class EvaluationItem(ABC):
    @abstractmethod
    def merge(self, other):
        """Merges another item from a different scene into this one.

        This is used to average metrics over scenes. Merges by taking a
        weighted average (by gt_count) of the metrics.
        """

    @abstractmethod
    def to_json(self) -> dict:
        return self.__dict__

    def __repr__(self):
        return pformat(self.to_json())
