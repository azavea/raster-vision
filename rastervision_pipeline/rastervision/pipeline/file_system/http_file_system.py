import os
import shutil
import urllib
import urllib.request
from datetime import datetime

from rastervision.pipeline.file_system import (FileSystem, NotReadableError,
                                               NotWritableError)
from urllib.parse import urlparse


class HttpFileSystem(FileSystem):
    """A FileSystem for downloading files over HTTP."""

    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        parsed_uri = urlparse(uri)
        return parsed_uri.scheme in ['http', 'https']

    @staticmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        try:
            response = urllib.request.urlopen(uri)
            if response.getcode() == 200:
                return int(response.headers['content-length']) > 0
            else:
                return False  # pragma: no cover
        except urllib.error.URLError:
            return False

    @staticmethod
    def read_str(uri: str) -> str:
        return HttpFileSystem.read_bytes(uri).decode('utf8')

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        with urllib.request.urlopen(uri) as req:
            return req.read()

    @staticmethod
    def write_str(uri: str, data: str) -> None:
        raise NotWritableError('Could not write {}'.format(uri))

    @staticmethod
    def write_bytes(uri: str, data: bytes) -> None:
        raise NotWritableError('Could not write {}'.format(uri))

    @staticmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str,
                    delete: bool = False) -> None:
        raise NotWritableError('Could not write {}'.format(dst_dir_uri))

    @staticmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str,
                      delete: bool = False) -> None:
        raise NotReadableError(
            'Cannot read directory from HTTP {}'.format(src_dir_uri))

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        raise NotWritableError('Could not write {}'.format(dst_uri))

    @staticmethod
    def copy_from(src_uri: str, dst_path: str) -> None:
        with urllib.request.urlopen(src_uri) as response:
            with open(dst_path, 'wb') as out_file:
                try:
                    shutil.copyfileobj(response, out_file)
                except Exception:  # pragma: no cover
                    raise NotReadableError('Could not read {}'.format(src_uri))

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 'http', parsed_uri.netloc,
                            parsed_uri.path[1:])
        # This function is expected to return something that is file path-like
        # (as opposed to directory-like),
        # so if the path ends with / we strip it off. This was motivated by
        # a URI that was a zxy tile schema that doesn't end in .png which is
        # parsed by urlparse into a path that ends in a /.
        if path.endswith('/'):
            path = path[:-1]
        return path

    @staticmethod
    def last_modified(uri: str) -> datetime:
        return None

    @staticmethod
    def list_paths(uri, suffix=None):  # pragma: no cover
        raise NotImplementedError()
