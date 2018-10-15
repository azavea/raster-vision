import os
import shutil
from threading import Timer
import logging

from google.protobuf import json_format

from rastervision.filesystem.filesystem import FileSystem
from rastervision.filesystem.filesystem import ProtobufParseException
from rastervision.filesystem.local_filesystem import make_dir

log = logging.getLogger(__name__)


def get_local_path(uri, download_dir, fs=None):
    """Convert a URI into a corresponding local path.

    If a uri is local, return it. If it's remote, we generate a path for it
    within download_dir. For an S3 path of form s3://<bucket>/<key>, the path
    is <download_dir>/s3/<bucket>/<key>.

    Args:
        uri: (string) URI of file
        download_dir: (string) path to directory
        fs: Optional FileSystem to use

    Returns:
        (string) a local path
    """
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    path = fs.local_path(uri, download_dir)

    return path


def sync_to_dir(src_dir_uri, dest_dir_uri, delete=False, fs=None):
    """Synchronize a local to a local or remote directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If delete is True, also delete
    files in the destination to match those in the source directory.

    Args:
        src_dir_uri: (string) URI of local source directory
        dest_dir_uri: (string) URI of destination directory
        delete: (bool)
        fs: Optional FileSystem to use for destination
    """
    if not fs:
        fs = FileSystem.get_file_system(dest_dir_uri, 'w')
    fs.sync_to_dir(src_dir_uri, dest_dir_uri, delete=delete)


def sync_from_dir(src_dir_uri, dest_dir_uri, delete=False, fs=None):
    """Synchronize a local or remote directory to a local directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If delete is True, also delete
    files in the destination to match those in the source directory.

    Args:
        src_dir_uri: (string) URI of source directory
        dest_dir_uri: (string) URI of local destination directory
        delete: (bool)
        fs: Optional FileSystem to use
    """
    if not fs:
        fs = FileSystem.get_file_system(src_dir_uri, 'r')
    fs.sync_from_dir(src_dir_uri, dest_dir_uri, delete=delete)


def start_sync(src_dir_uri, dest_dir_uri, sync_interval=600, fs=None):
    """Start syncing a directory on a schedule.

    Calls sync_to_dir on a schedule.

    Args:
        src_dir_uri: (string) Path of the local source directory
        dest_dir_uri: (string) URI of destination directory
        sync_interval: (int) period in seconds for syncing
        fs:  Optional FileSystem to use
    """

    def _sync_dir():
        log.info('Syncing {} to {}...'.format(src_dir_uri, dest_dir_uri))
        sync_to_dir(src_dir_uri, dest_dir_uri, delete=False, fs=fs)

    class SyncThread:
        def __init__(self):
            thread = Timer(sync_interval, _sync_dir)
            thread.daemon = True
            thread.start()
            self.thread = thread

        def __enter__(self):
            return self.thread

        def __exit__(self, type, value, traceback):
            self.thread.cancel()

    return SyncThread()


def download_if_needed(uri, download_dir, fs=None):
    """Download a file into a directory if it's remote.

    If uri is local, there is no need to download the file.

    Args:
        uri: (string) URI of file
        download_dir: (string) local directory to download file into
        fs: Optional FileSystem to use.

    Returns:
        (string) path to local file

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
        log.info('Downloading {} to {}'.format(uri, path))

    fs.copy_from(uri, path)

    return path


def download_or_copy(uri, target_dir, fs=None):
    """Downloads or copies a file to a directory

    Args:
       uri: (string) URI of file
       target_dir: (string) local directory to copy file to
       fs: Optional FileSystem to use
    """
    local_path = download_if_needed(uri, target_dir, fs=fs)
    shutil.copy(local_path, target_dir)
    return local_path


def file_exists(uri, fs=None):
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.file_exists(uri)


def list_paths(uri, ext='', fs=None):
    if uri is None:
        return None

    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')

    return fs.list_paths(uri, ext=ext)


def upload_or_copy(src_path, dst_uri, fs=None):
    """Upload a file if the destination is remote.

    If dst_uri is local, the file is copied.

    Args:
        src_path: (string) path to source file
        dst_uri: (string) URI of destination for file
        fs: Optional FileSystem to use
    Raises:
        NotWritableError if URI cannot be written to
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


def file_to_str(uri, fs=None):
    """Download contents of text file into a string.

    Args:
        uri: (string) URI of file
        fs: Optional FileSystem to use

    Returns:
        (string) with contents of text file

    Raises:
        NotReadableError if URI cannot be read from
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.read_str(uri)


def str_to_file(content_str, uri, fs=None):
    """Writes string to text file.

    Args:
        content_str: string to write
        uri: (string) URI of file to write
        fs: Optional FileSystem to use

    Raise:
        NotWritableError if file_uri cannot be written
    """
    if not fs:
        fs = FileSystem.get_file_system(uri, 'r')
    return fs.write_str(uri, content_str)


def load_json_config(uri, message, fs=None):
    """Load a JSON-formatted protobuf config file.

    Args:
        uri: (string) URI of config file
        message: (google.protobuf.message.Message) empty protobuf message of
            to load the config into. The type needs to match the content of
            uri.
        fs: Optional FileSystem to use.

    Returns:
        the same message passed as input with fields filled in from uri

    Raises:
        ProtobufParseException if uri cannot be parsed
    """
    try:
        return json_format.Parse(file_to_str(uri, fs=fs), message)
    except json_format.ParseError as e:
        error_msg = ('Problem parsing protobuf file {}. '.format(uri) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg) from e


def save_json_config(message, uri, fs=None):
    """Save a protobuf object to a JSON file.

    Args:
        message: (google.protobuf.message.Message) protobuf message
        uri: (string) URI of JSON file to write message to
        fs: Optional FileSystem to use

    Raises:
        NotWritableError if uri cannot be written
    """
    json_str = json_format.MessageToJson(message)
    str_to_file(json_str, uri, fs=fs)
