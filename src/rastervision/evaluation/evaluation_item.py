from abc import (ABC, abstractmethod)


class EvaluationItem(ABC):
    @abstractmethod
    def merge(self, other):
        """Merges another item from a different scene into this one.

        This is used to average metrics over scenes. Merges by taking a
        weighted average (by gt_count) of the metrics.
        """
        pass

    @abstractmethod
    def to_json(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_json())
