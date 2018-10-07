import os
import shutil
import urllib
import urllib.request
from datetime import datetime

from rastervision.filesystem import (FileSystem, NotReadableError,
                                     NotWritableError)
from urllib.parse import urlparse


class HttpFileSystem(FileSystem):
    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        if mode == 'r':
            parsed_uri = urlparse(uri)
            return parsed_uri.scheme in ['http', 'https']
        else:
            return False

    @staticmethod
    def file_exists(uri: str) -> bool:
        try:
            response = urllib.request.urlopen(uri)
            if response.getcode() == 200:
                return int(response.headers['content-length']) > 0
            else:
                return False
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
    def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                    delete: bool = False) -> None:
        raise NotWritableError('Could not write {}'.format(dest_dir_uri))

    @staticmethod
    def sync_from_dir(src_dir_uri: str,
                      dest_dir_uri: str,
                      delete: bool = False) -> None:
        raise NotReadableError(
            'Cannot read directory from HTTP {}'.format(src_dir_uri))

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        raise NotWritableError('Could not write {}'.format(dst_uri))

    @staticmethod
    def copy_from(uri: str, path: str) -> None:
        with urllib.request.urlopen(uri) as response:
            with open(path, 'wb') as out_file:
                try:
                    shutil.copyfileobj(response, out_file)
                except Exception:
                    raise NotReadableError('Could not read {}'.format(uri))

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        parsed_uri = urlparse(uri)
        path = os.path.join(download_dir, 'http', parsed_uri.netloc,
                            parsed_uri.path[1:])
        return path

    @staticmethod
    def last_modified(uri: str) -> datetime:
        return None

    @staticmethod
    def list_paths(uri, suffix=None):
        raise NotImplementedError()
