from abc import (ABC, abstractmethod)
from typing import Union


class FileSystem(ABC):

    @staticmethod
    @abstractmethod
    def matches_uri(uri: str) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def stat(uri: str) -> Union[int, None]:
        pass

    @staticmethod
    @abstractmethod
    def open(uri: str):
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def read(self) -> bytearray:
        pass

    @abstractmethod
    def write(self, data: bytearray) -> int:
        pass

    @abstractmethod
    def seek(self, offset: int, whence: int) -> int:
        pass

    @abstractmethod
    def tell(self) -> int:
        pass

