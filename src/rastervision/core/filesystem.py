from abc import (ABC, abstractmethod, staticmethod)


class FileSystem(ABC):

    @staticmethod
    @abstractmethod
    def matches_uri(uri):
        pass

    @staticmethod
    @abstractmethod
    def stat(uri):
        pass

    @staticmethod
    @abstractmethod
    def open(uri):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def lseek(self):
        pass

    @abstractmethod
    def tell(self, uri):
        pass

