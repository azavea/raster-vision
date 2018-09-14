from abc import (ABC, abstractmethod)


class FileSystem(ABC):
    @staticmethod
    @abstractmethod
    def matches_uri(uri: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def file_exists(uri: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def read(self, uri: str) -> bytearray:
        pass

    @staticmethod
    @abstractmethod
    def write(self, uri: str, data: bytearray) -> int:
        pass
