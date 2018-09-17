from abc import (ABC, abstractmethod)
from typing import Union


class NotReadableError(Exception):
    pass


class NotWritableError(Exception):
    pass


class ProtobufParseException(Exception):
    pass

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
    def read_str(uri: str) -> str:
        pass

    @staticmethod
    @abstractmethod
    def read_bytes(uri: str) -> bytes:
        pass

    @staticmethod
    @abstractmethod
    def write_str(uri: str, data: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def write_bytes(uri: str, data: bytes) -> None:
        pass

    @staticmethod
    @abstractmethod
    def sync_dir(src_dir_uri: str, dest_dir_uri: str, delete: bool=False) -> None:
        pass

    @staticmethod
    @abstractmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def copy_from(uri: str, path: str) -> None:
        pass

    @staticmethod
    @abstractmethod
    def local_path(uri: str, download_dir: str) -> None:
        pass

