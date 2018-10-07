import os
import shutil
from datetime import datetime, timezone
import glob

from rastervision.filesystem import (FileSystem, NotReadableError)


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
        directory = os.path.abspath(os.path.dirname(path))

    if force_empty and os.path.isdir(directory):
        shutil.rmtree(directory)

    os.makedirs(directory, exist_ok=True)

    is_empty = len(os.listdir(directory)) == 0
    if check_empty and not is_empty:
        raise ValueError(
            '{} needs to be an empty directory!'.format(directory))


class LocalFileSystem(FileSystem):
    @staticmethod
    def matches_uri(uri: str, mode: str) -> bool:
        return True

    @staticmethod
    def file_exists(uri: str) -> bool:
        return os.path.isfile(uri)

    @staticmethod
    def read_str(file_uri: str) -> str:
        if not os.path.isfile(file_uri):
            raise NotReadableError('Could not read {}'.format(file_uri))
        with open(file_uri, 'r') as file_buffer:
            return file_buffer.read()

    @staticmethod
    def read_bytes(file_uri: str) -> bytes:
        if not os.path.isfile(file_uri):
            raise NotReadableError('Could not read {}'.format(file_uri))
        with open(file_uri, 'rb') as file_buffer:
            return file_buffer.read()

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
    def sync_from_dir(src_dir_uri: str,
                      dest_dir_uri: str,
                      delete: bool = False) -> None:
        if src_dir_uri == dest_dir_uri:
            return

        if delete:
            shutil.rmtree(dest_dir_uri)

        # https://stackoverflow.com/a/15824216/841563
        def recursive_overwrite(src, dest, ignore=None):
            if os.path.isdir(src):
                if not os.path.isdir(dest):
                    os.makedirs(dest)
                    files = os.listdir(src)
                    if ignore is not None:
                        ignored = ignore(src, files)
                    else:
                        ignored = set()
                        for f in files:
                            if f not in ignored:
                                recursive_overwrite(
                                    os.path.join(src, f), os.path.join(
                                        dest, f), ignore)
            else:
                shutil.copyfile(src, dest)

        recursive_overwrite(src_dir_uri, dest_dir_uri)

    @staticmethod
    def sync_to_dir(src_dir_uri: str, dest_dir_uri: str,
                    delete: bool = False) -> None:
        LocalFileSystem.sync_from_dir(src_dir_uri, dest_dir_uri, delete)

    @staticmethod
    def copy_to(src_path: str, dst_uri: str) -> None:
        if src_path != dst_uri:
            make_dir(dst_uri, use_dirname=True)
            shutil.copyfile(src_path, dst_uri)

    @staticmethod
    def copy_from(uri: str, path: str) -> None:
        not_found = not os.path.isfile(path)
        if not_found:
            raise NotReadableError('Could not read {}'.format(uri))

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
