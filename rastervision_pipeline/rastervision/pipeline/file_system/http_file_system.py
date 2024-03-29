from typing import ContextManager
import os
import io
import shutil
import urllib
import urllib.request
from urllib.parse import urlparse
import requests
from datetime import datetime
from functools import partial

from tqdm.auto import tqdm

from rastervision.pipeline.file_system import (FileSystem, NotReadableError,
                                               NotWritableError)


def get_file_obj(uri: str, with_progress: bool = True,
                 **kwargs) -> ContextManager:
    """Returns a context manager for a file-like object that supports buffered
    reads. If with_progress is True, wraps the read() method of the object in
    a function that updates a tqdm progress bar.

    Usage:

    .. code-block:: python

        with get_file_obj(uri) as f:
            ...

    Adapted from https://stackoverflow.com/a/63831344/5908685.
    """
    r = requests.get(uri, stream=True, allow_redirects=True, **kwargs)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(
            f'Request to {uri} returned status code {r.status_code}')
    file_obj = r.raw
    # Decompress if needed
    file_obj.read = partial(file_obj.read, decode_content=True)
    if not with_progress:
        return file_obj
    file_size = int(r.headers.get('Content-Length', 0))
    desc = '(Unknown total file size)' if file_size == 0 else ''
    # put a wrapper around file_obj's read() method that updates the
    # progress bar
    file_obj_wrapped = tqdm.wrapattr(
        file_obj,
        'read',
        total=file_size,
        desc=desc,
        bytes=True,
        mininterval=0.5,
        delay=5)
    return file_obj_wrapped


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
            return response.getcode() == 200
        except urllib.error.URLError:
            return False

    @staticmethod
    def read_str(uri: str) -> str:
        return HttpFileSystem.read_bytes(uri).decode('utf8')

    @staticmethod
    def read_bytes(uri: str) -> bytes:
        with get_file_obj(uri) as in_file, io.BytesIO() as write_buffer:
            shutil.copyfileobj(in_file, write_buffer)
            return write_buffer.getvalue()

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
    def copy_from(src_uri: str, dst_path: str, **kwargs) -> None:
        with get_file_obj(src_uri, **kwargs) as in_file:
            with open(dst_path, 'wb') as out_file:
                shutil.copyfileobj(in_file, out_file)

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
    def list_paths(uri, suffix=None):
        raise NotImplementedError()
