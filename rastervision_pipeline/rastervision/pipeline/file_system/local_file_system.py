import os
import io
import shutil
from datetime import datetime, timezone
import glob

from tqdm.auto import tqdm

from rastervision.pipeline.file_system import (FileSystem, NotReadableError)


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
    """Make a local directory.

    Args:
        path: path to directory
        check_empty: if True, check that directory is empty
        force_empty: if True, delete files if necessary to make directory
            empty
        use_dirname: if True, use the the parent directory as path

    Raises:
        ValueError if check_empty is True and directory is not empty
    """
    directory = path
    if use_dirname:
        directory = os.path.abspath(os.path.dirname(path))

    if force_empty and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    if check_empty:
        with os.scandir(directory) as it:
            if any(it):
                raise ValueError(
                    f'{directory} needs to be an empty directory!')


def progressbar(file_obj, method: str, size: int, desc: str):
    return tqdm.wrapattr(
        file_obj,
        method,
        total=size,
        desc=desc,
        bytes=True,
        mininterval=0.5,
        delay=5)


class LocalFileSystem(FileSystem):
    """A FileSystem for interacting with the local file system."""

    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        return True

    @staticmethod
    def file_exists(uri: str, include_dir: bool = True) -> bool:
        return (os.path.isfile(uri) or (include_dir and os.path.isdir(uri)))

    @staticmethod
    def read_str(file_uri: str) -> str:
        if not os.path.isfile(file_uri):
            raise NotReadableError(f'Could not read {file_uri}')

        file_size = os.path.getsize(file_uri)
        with open(file_uri, 'r') as in_file, io.StringIO() as str_buffer:
            with progressbar(
                    in_file, 'read', file_size, desc='Reading file') as bar:
                shutil.copyfileobj(bar, str_buffer)
            return str_buffer.getvalue()

    @staticmethod
    def read_bytes(file_uri: str) -> bytes:
        if not os.path.isfile(file_uri):
            raise NotReadableError(f'Could not read {file_uri}')

        file_size = os.path.getsize(file_uri)
        with open(file_uri, 'rb') as in_file, io.BytesIO() as byte_buffer:
            with progressbar(
                    in_file, 'read', file_size, desc='Reading file') as bar:
                shutil.copyfileobj(bar, byte_buffer)
            return byte_buffer.getvalue()

    @staticmethod
    def write_str(file_uri: str, data: str) -> None:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'w') as content_file:
            content_file.write(data)

    @staticmethod
    def write_bytes(file_uri: str, data: bytes) -> None:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'wb') as content_file:
            content_file.write(data)

    @staticmethod
    def sync_from_dir(src_dir_uri: str, dst_dir: str,
                      delete: bool = False) -> None:
        if src_dir_uri == dst_dir:
            return

        if delete:
            shutil.rmtree(dst_dir)

        # https://stackoverflow.com/a/15824216/841563
        def recursive_overwrite(src, dest):
            if os.path.isdir(src):
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                    for entry in os.scandir(src):
                        recursive_overwrite(entry.path,
                                            os.path.join(dest, entry.name))
            else:
                shutil.copyfile(src, dest)

        recursive_overwrite(src_dir_uri, dst_dir)

    @staticmethod
    def sync_to_dir(src_dir: str, dst_dir_uri: str,
                    delete: bool = False) -> None:
        LocalFileSystem.sync_from_dir(src_dir, dst_dir_uri, delete)

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        if src_path != dst_uri:
            make_dir(dst_uri, use_dirname=True)
            shutil.copyfile(src_path, dst_uri)

    @staticmethod
    def copy_from(src_uri: str, dst_path: str) -> None:
        LocalFileSystem.copy_to(src_uri, dst_path)

    @staticmethod
    def local_path(uri: str, download_dir: str) -> None:
        path = uri
        return path

    @staticmethod
    def last_modified(uri: str) -> datetime:
        local_last_modified = datetime.utcfromtimestamp(os.path.getmtime(uri))
        return local_last_modified.replace(tzinfo=timezone.utc)

    @staticmethod
    def list_paths(uri, ext=None):
        if ext is None:
            ext = ''
        return glob.glob(os.path.join(uri, '*' + ext))
