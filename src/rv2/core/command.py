from abc import ABC, abstractmethod


class Command(ABC):
    @abstractmethod
    def run(self):
        pass
