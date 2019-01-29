from abc import (ABC, abstractmethod)
from datetime import datetime

import rastervision as rv


class NotReadableError(Exception):
    pass


class NotWritableError(Exception):
    pass


class ProtobufParseException(Exception):
    pass


class FileSystem(ABC):
    @staticmethod
    def get_file_system(uri, mode='r'):
        return rv._registry.get_file_system(uri, mode)

    @staticmethod
    @abstractmethod
    def matches_uri(uri: str, mode: str) -> bool:
        """Returns True if this FileSystem should be used
        for the given URI under the given mode.

        Mode can be 'r' or 'w'
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        """Check if a  file exists.
        Args:
          uri: The URI to check
          include_dir: Include directories in check, if this filesystem
                       supports directory reads. Otherwise only
                       return true if a single file exists at the URI.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def read_str(uri: str) -> str:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def read_bytes(uri: str) -> bytes:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def write_str(uri: str, data: str) -> None:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def write_bytes(uri: str, data: bytes) -> None:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                    delete: bool = False) -> None:
        """Syncs a local directory to a destination.

        Arguments:
           - src_dir_uri: Source directory to sync from. Must be a local directoy.
           - dest_dir_uri: A destination that can be sync to by this FileSystem.
           - delete: True if the destination should be deleted first. Defaults to False.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def sync_from_dir(src_dir_uri: str,
                      dest_dir_uri: str,
                      delete: bool = False) -> None:
        """Syncs a local directory to a destination.

        Arguments:
           - src_dir_uri: Source directory to sync from. Must be a local directoy.
           - dest_dir_uri: A destination that can be sync to by this FileSystem.
           - delete: True if the destination should be deleted first. Defaults to False.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def copy_from(uri: str, path: str) -> None:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def local_path(uri: str, download_dir: str) -> None:
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def last_modified(uri: str) -> datetime:
        """Returns the last modified  date in UTC of this URI,
        or None if this FileSystem does not support this operation.
        """
        pass  # pragma: no cover

    @staticmethod
    @abstractmethod
    def list_paths(uri, ext=None):
        pass  # pragma: no cover
