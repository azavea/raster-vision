import os
import shutil

from rastervision.filesystem.filesystem import *
from typing import Union


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
    """Make a local directory.

    Args:
        path: path to directory
        check_empty: if True, check that directory is empty
        force_empty: if True, delete files if necessary to make directory
            empty
        use_dirname: if path is a file, use the the parent directory as path

    Raises:
        ValueError if check_empty is True and directory is not empty
    """
    directory = path
    if use_dirname:
        directory = os.path.dirname(path)

    if force_empty and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    is_empty = len(os.listdir(directory)) == 0
    if check_empty and not is_empty:
        raise ValueError(
            '{} needs to be an empty directory!'.format(directory))

class LocalFileSystem(FileSystem):
    def matches_uri(uri: str) -> bool:
        return True

    def file_exists(uri: str) -> bool:
        return os.path.isfile(uri)

    def read_str(file_uri: str) -> str:
        if not os.path.isfile(file_uri):
            raise NotReadableError('Could not read {}'.format(file_uri))
        with open(file_uri, 'r') as file_buffer:
            return file_buffer.read()

    def read_bytes(file_uri: str) -> bytes:
        if not os.path.isfile(file_uri):
            raise NotReadableError('Could not read {}'.format(file_uri))
        with open(file_uri, 'rb') as file_buffer:
            return file_buffer.read()

    def write_str(file_uri: str, data: str) -> None:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'w') as content_file:
            content_file.write(data)

    def write_bytes(file_uri: str, data: bytes) -> None:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'wb') as content_file:
            content_file.write(data)

    def sync_dir(src_dir_uri: str, dest_dir_uri: str, delete: bool=False) -> None:
        pass # XXX

    def copy_to(src_path: str, dst_uri: str) -> None:
        if src_path != dst_uri:
            make_dir(dst_uri, use_dirname=True)
            shutil.copyfile(src_path, dst_uri)

    def copy_from(uri: str, path: str) -> None:
        not_found = not os.path.isfile(path)
        if not_found:
            raise NotReadableError('Could not read {}'.format(uri))

    def local_path(uri: str, download_dir: str) -> None:
        path = uri
        return path
