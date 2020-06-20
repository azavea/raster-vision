import os
from os.path import join
import shutil
import gzip
from threading import Timer
import time
import logging
import json
import zipfile
from typing import Optional, List

from rastervision.pipeline.file_system import FileSystem
from rastervision.pipeline.file_system.local_file_system import make_dir

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
                       download_dir: str,
                       fs: Optional[FileSystem] = None) -> str:
    """Download a file into a directory if it's remote.

    If uri is local, there is no need to download the file.

    Args:
        uri: URI of file
        download_dir: local directory to download file into
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            uri

    Returns:
        path to local file

    Raises:
        NotReadableError if URI cannot be read from
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')

    path = get_local_path(uri, download_dir, fs=fs)
    make_dir(path, use_dirname=True)

    if path != uri:
        log.debug('Downloading {} to {}'.format(uri, path))

    fs.copy_from(uri, path)

    return path


def download_or_copy(uri, target_dir, fs=None) -> str:
    """Downloads or copies a file to a directory.

    Downloads or copies URI into target_dir.

    Args:
        uri: URI of file
        target_dir: local directory to download or copy file to
        fs: if supplied, use fs instead of automatically chosen FileSystem for
            uri

    Returns:
        the local path of file
    """
    local_path = download_if_needed(uri, target_dir, fs=fs)
    shutil.copy(local_path, target_dir)
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
                   fs: Optional[FileSystem] = None) -> List[str]:
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


def get_cached_file(cache_dir: str, uri: str) -> str:
    """Download a file and unzip it using a cache.

    This downloads a file if it isn't already in the cache, and unzips
    the file using gunzip if it hasn't already been unzipped (and the uri
    has a .gz suffix).

    Args:
        cache_dir: dir to use for cache directory
        uri: URI of a file that can be opened by a supported RV file system

    Returns:
        path of the (downloaded and unzipped) cached file
    """
    # Only download if it isn't in the cache.
    path = get_local_path(uri, cache_dir)
    if not os.path.isfile(path):
        path = download_if_needed(uri, cache_dir)

    # Unzip if .gz file
    if path.endswith('.gz'):
        # If local URI, then make ungz_path in temp cache, so it isn't unzipped
        # alongside the original file.
        if os.path.isfile(uri):
            ungz_path = os.path.join(cache_dir, path)[:-3]
        else:
            ungz_path = path[:-3]

        # Check to see if it is already unzipped before unzipping.
        if not os.path.isfile(ungz_path):
            with gzip.open(path, 'rb') as f_in:
                with open(ungz_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        path = ungz_path

    return path


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
        for dirpath, dirnames, filenames in os.walk(dir):
            for fn in filenames:
                ziph.write(join(dirpath, fn), join(dirpath[len(dir):], fn))


def unzip(zip_path: str, target_dir: str):
    """Unzip contents of zip file at zip_path into target_dir.

    Creates target_dir if needed.
    """
    make_dir(target_dir)
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()
