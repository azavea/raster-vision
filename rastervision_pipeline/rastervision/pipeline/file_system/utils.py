import os
from os.path import join, normpath, relpath
import shutil
from threading import Timer
import time
import logging
import json
import zipfile
from typing import TYPE_CHECKING, Optional, List

from tqdm.auto import tqdm

from rastervision.pipeline import rv_config_ as rv_config
from rastervision.pipeline.file_system import FileSystem
from rastervision.pipeline.file_system.local_file_system import (
    LocalFileSystem, make_dir)

if TYPE_CHECKING:
    from tempfile import TemporaryDirectory

log = logging.getLogger(__name__)


def get_local_path(uri: str,
                   download_dir: str,
                   fs: Optional[FileSystem] = None) -> str:
    """Return the path where a local copy of URI should be stored.

    If URI is local, return it. If it's remote, we generate a path for it
    within download_dir.

    Args:
        uri: the URI of the file to be copied
        download_dir: path of the local directory in which files should
            be copied
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            URI

    Returns:
        a local path
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    path = fs.local_path(uri, download_dir)

    return path


def sync_to_dir(src_dir: str,
                dst_dir_uri: str,
                delete: bool = False,
                fs: Optional[FileSystem] = None):
    """Synchronize a local source directory to destination directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If FileSystem is remote, this involves
    uploading.

    Args:
        src_dir: path of local source directory
        dst_dir_uri: URI of destination directory
        delete: if True, delete files in the destination to match those in the
            source directory
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            dst_dir_uri
    """
    if not fs:
        fs = FileSystem.get_file_system(dst_dir_uri, 'w')
    fs.sync_to_dir(src_dir, dst_dir_uri, delete=delete)


def sync_from_dir(src_dir_uri: str,
                  dst_dir: str,
                  delete: bool = False,
                  fs: Optional[FileSystem] = None):
    """Synchronize a source directory to local destination directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If FileSystem is remote, this involves
    downloading.

    Args:
        src_dir_uri: URI of source directory
        dst_dir: path of local destination directory
        delete: if True, delete files in the destination to match those in the
            source directory
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            dst_dir_uri
    """
    if not fs:
        fs = FileSystem.get_file_system(src_dir_uri, 'r')
    fs.sync_from_dir(src_dir_uri, dst_dir, delete=delete)


def start_sync(src_dir: str,
               dst_dir_uri: str,
               sync_interval: int = 600,
               fs: Optional[FileSystem] = None):  # pragma: no cover
    """Repeatedly sync a local source directory to a destination on a schedule.

    Calls sync_to_dir on a schedule.

    Args:
        src_dir: path of the local source directory
        dst_dir_uri: URI of destination directory
        sync_interval: period in seconds for syncing
        fs: if supplied, use fs instead of automatically chosen FileSystem
    """

    def _sync_dir():
        while True:
            time.sleep(sync_interval)
            log.info('Syncing {} to {}...'.format(src_dir, dst_dir_uri))
            sync_to_dir(src_dir, dst_dir_uri, delete=False, fs=fs)

    class SyncThread:
        def __init__(self):
            thread = Timer(0.68, _sync_dir)
            thread.daemon = True
            thread.start()
            self.thread = thread

        def __enter__(self):
            return self.thread

        def __exit__(self, type, value, traceback):
            self.thread.cancel()

    return SyncThread()


def download_if_needed(uri: str,
                       download_dir: Optional[str] = None,
                       fs: Optional[FileSystem] = None,
                       use_cache: bool = True) -> str:
    """Download a file into a directory if it's remote.

    If uri is local, there is no need to download the file.

    Args:
        uri (str): URI of file to download.
        download_dir (Optional[str], optional): Local directory to download
            file into. If None, the file will be downloaded to
            cache dir as defined by RVConfig. Defaults to None.
        fs (Optional[FileSystem], optional): If provided, use fs instead of
            the automatically chosen FileSystem for uri. Defaults to None.
        use_cache (bool, optional): If False and the file is remote, download
            it regardless of whether it exists in cache. Defaults to True.

    Returns:
        str: Path to local file.

    Raises:
        NotReadableError if URI cannot be read from
    """
    if download_dir is None:
        download_dir = rv_config.get_cache_dir()

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')

    local_path = get_local_path(uri, download_dir, fs=fs)
    if local_path == uri:
        return local_path

    if use_cache and file_exists(local_path, include_dir=False):
        log.info(f'Using cached file {local_path}.')
        return local_path

    log.info(f'Downloading {uri} to {local_path}...')
    make_dir(local_path, use_dirname=True)
    fs.copy_from(uri, local_path)

    return local_path


def download_or_copy(uri: str,
                     target_dir: str,
                     delete_tmp: bool = False,
                     fs: Optional[FileSystem] = None) -> str:
    """Downloads or copies a file to a directory.

    Downloads or copies URI into target_dir.

    Args:
        uri: URI of file.
        target_dir: Local directory to download or copy file to.
        delete_tmp: Delete temporary download dir after copying file.
        fs: If supplied, use fs instead of automatically chosen FileSystem for
            uri.

    Returns:
        the local path of file
    """
    target_dir = normpath(target_dir)
    local_path = download_if_needed(uri, target_dir, fs=fs)
    shutil.copy(local_path, target_dir)

    if delete_tmp and not is_local(uri):
        dl_dirname = normpath(relpath(local_path, target_dir)).split(os.sep)[0]
        dl_dir = join(target_dir, dl_dirname)
        shutil.rmtree(dl_dir)
    return local_path


def file_exists(uri, fs=None, include_dir=True) -> bool:
    """Check if file exists.

    Args:
        uri: URI of file
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            uri
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.file_exists(uri, include_dir)


def list_paths(uri: str, ext: str = '',
               fs: Optional[FileSystem] = None) -> List[str]:
    """List paths rooted at URI.

    Optionally only includes paths with a certain file extension.

    Args:
        uri: the URI of a directory
        ext: the optional file extension to filter by
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            uri
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')

    return fs.list_paths(uri, ext=ext)


def upload_or_copy(src_path: str,
                   dst_uri: str,
                   fs: Optional[FileSystem] = None) -> None:
    """Upload or copy a file.

    If dst_uri is local, the file is copied. Otherwise, it is uploaded.

    Args:
        src_path: path to source file
        dst_uri: URI of destination for file
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            dst_uri

    Raises:
        NotWritableError if dst_uri cannot be written to
    """
    if dst_uri is None:
        return

    if not (os.path.isfile(src_path) or os.path.isdir(src_path)):
        raise Exception('{} does not exist.'.format(src_path))

    if not src_path == dst_uri:
        log.info('Uploading {} to {}'.format(src_path, dst_uri))

    if not fs:
        fs = FileSystem.get_file_system(dst_uri, 'w')
    fs.copy_to(src_path, dst_uri)


def file_to_str(uri: str, fs: Optional[FileSystem] = None) -> str:
    """Load contents of text file into a string.

    Args:
        uri: URI of file
        fs: if supplied, use fs instead of automatically chosen FileSystem

    Returns:
        contents of text file

    Raises:
        NotReadableError if URI cannot be read
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.read_str(uri)


def str_to_file(content_str: str, uri: str, fs: Optional[FileSystem] = None):
    """Writes string to text file.

    Args:
        content_str: string to write
        uri: URI of file to write
        fs: if supplied, use fs instead of automatically chosen FileSystem

    Raise:
        NotWritableError if uri cannot be written
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.write_str(uri, content_str)


def file_to_json(uri: str) -> dict:
    """Return JSON dict based on file at uri."""
    return json.loads(file_to_str(uri))


def json_to_file(content_dict: dict, uri: str):
    """Upload JSON file to uri based on content_dict."""
    str_to_file(json.dumps(content_dict), uri)


def zipdir(dir: str, zip_path: str):
    """Save a zip file with contents of directory.

    Contents of directory will be at root of zip file.

    Args:
        dir: directory to zip
        zip_path: path to zip file to create
    """
    make_dir(zip_path, use_dirname=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as ziph:
        with tqdm(desc='Zipping', delay=5) as bar:
            for dirpath, dirnames, filenames in os.walk(dir):
                for fn in filenames:
                    bar.set_postfix_str(fn)
                    src = join(dirpath, fn)
                    dst = join(dirpath[len(dir):], fn)
                    ziph.write(src, dst)
                    bar.update(1)


def unzip(zip_path: str, target_dir: str):
    """Unzip contents of zip file at zip_path into target_dir.

    Creates target_dir if needed.
    """
    make_dir(target_dir)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(target_dir)


def is_local(uri: str) -> bool:
    return FileSystem.get_file_system(uri) == LocalFileSystem


def is_archive(uri: str) -> bool:
    """Check if the URI's extension represents an archived file."""
    formats = sum((fmts for _, fmts, _ in shutil.get_unpack_formats()), [])
    return any(uri.endswith(fmt) for fmt in formats)


def extract(uri: str,
            target_dir: Optional[str] = None,
            download_dir: Optional[str] = None) -> str:
    """Extract a compressed file."""
    if target_dir is None:
        target_dir = rv_config.get_cache_dir()
    make_dir(target_dir)
    local_path = download_if_needed(uri, download_dir)
    shutil.unpack_archive(local_path, target_dir)
    return target_dir


def get_tmp_dir() -> 'TemporaryDirectory':
    """Return temporary directory given by the RVConfig.

    Returns:
        TemporaryDirectory: A context manager.
    """
    return rv_config.get_tmp_dir()
