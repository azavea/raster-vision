from abc import (ABC, abstractmethod)
from datetime import datetime
from typing import Optional, List

from rastervision.pipeline import registry_ as registry


class NotReadableError(Exception):
    """Exception raised when files are not readable."""


class NotWritableError(Exception):
    """Exception raised when files are not writable."""


class FileSystem(ABC):
    """Abstraction for a local or remote file system.

    This can be subclassed to handle different cloud storage providers, etc.
    """

    @staticmethod
    def get_file_system(uri: str, mode: str = 'r') -> 'FileSystem':
        """Return FileSystem that should be used for the given URI/mode pair.

        Args:
            uri: URI of file
            mode: mode to open file in, 'r' or 'w'
        """
        return registry.get_file_system(uri, mode)

    @staticmethod
    @abstractmethod
    def matches_uri(uri: str, mode: str) -> bool:
        """Returns True if this FS can be used for the given URI/mode pair.

        Args:
            uri: URI of file
            mode: mode to open file in, 'r' or 'w'
        """

    @staticmethod
    @abstractmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        """Check if a file exists.

        Args:
          uri: The URI to check
          include_dir: Include directories in check, if this file_system
            supports directory reads. Otherwise only return true if a single
            file exists at the URI.
        """

    @staticmethod
    @abstractmethod
    def read_str(uri: str) -> str:
        """Read contents of URI to a string."""

    @staticmethod
    @abstractmethod
    def read_bytes(uri: str) -> bytes:
        """Read contents of URI to bytes."""

    @staticmethod
    @abstractmethod
    def write_str(uri: str, data: str):
        """Write string in data to URI."""

    @staticmethod
    @abstractmethod
    def write_bytes(uri: str, data: bytes):
        """Write bytes in data to URI."""

    @staticmethod
    @abstractmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str, delete: bool = False):
        """Syncs a local source directory to a destination directory.

        If the FileSystem is remote, this involves uploading.

        Args:
            src_dir: local source directory to sync from
            dst_dir_uri: A destination directory that can be synced to by this
                FileSystem
            delete: True if the destination should be deleted first.
        """

    @staticmethod
    @abstractmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str, delete: bool = False):
        """Syncs a source directory to a local destination directory.

        If the FileSystem is remote, this involves downloading.

        Args:
            src_dir_uri: source directory that can be synced from by this FileSystem
            dst_dir: A local destination directory
            delete: True if the destination should be deleted first.
        """

    @staticmethod
    @abstractmethod
    def copy_to(src_path: str, dst_uri: str):
        """Copy a local source file to a destination.

        If the FileSystem is remote, this involves uploading.

        Args:
            src_path: local path to source file
            dst_uri: uri of destination that can be copied to by this FileSystem
        """

    @staticmethod
    @abstractmethod
    def copy_from(src_uri: str, dst_path: str):
        """Copy a source file to a local destination.

        If the FileSystem is remote, this involves downloading.

        Args:
            src_uri: uri of source that can be copied from by this FileSystem
            dst_path: local path to destination file
        """

    @staticmethod
    @abstractmethod
    def local_path(uri: str, download_dir: str) -> str:
        """Return the path where a local copy should be stored.

        Args:
            uri: the URI of the file to be copied
            download_dir: path of the local directory in which files should
                be copied
        """

    @staticmethod
    @abstractmethod
    def last_modified(uri: str) -> Optional[datetime]:
        """Get the last modified date of a file.

        Args:
            uri: the URI of the file

        Returns:
            the last modified date in UTC of a file or None if this FileSystem
            does not support this operation.
        """

    @staticmethod
    @abstractmethod
    def list_paths(uri: str, ext: Optional[str] = None) -> List[str]:
        """List paths rooted at URI.

        Optionally only includes paths with a certain file extension.

        Args:
            uri: the URI of a directory
            ext: the optional file extension to filter by
        """
