import rastervision as rv
import io
import os
import urllib
from urllib.parse import urlparse
import urllib.request
import subprocess
import tempfile
from threading import Timer
from pathlib import Path

from google.protobuf import json_format

from rastervision.filesystem.filesystem import (NotReadableError, NotWritableError, ProtobufParseException)
from rastervision.filesystem.filesystem import FileSystem
from rastervision.filesystem.local_filesystem import make_dir


def get_local_path(uri, download_dir):
    """Convert a URI into a corresponding local path.

    If a uri is local, return it. If it's remote, we generate a path for it
    within download_dir. For an S3 path of form s3://<bucket>/<key>, the path
    is <download_dir>/s3/<bucket>/<key>.

    Args:
        uri: (string) URI of file
        download_dir: (string) path to directory

    Returns:
        (string) a local path
    """
    if uri is None:
        return None

    fs = FileSystem.get_file_system(uri)
    path = fs.local_path(uri, download_dir)

    return path


def sync_dir(src_dir_uri, dest_dir_uri, delete=False):
    """Synchronize a local and remote directory.

    Transfers files from source to destination directories so that the
    destination has all the source files. If delete is True, also delete
    files in the destination to match those in the source directory.

    Args:
        src_dir_uri: (string) URI of source directory
        dest_dir_uri: (string) URI of destination directory
        delete: (bool)
    """
    fs = rv._registry.get_file(dest_dir_uri)
    fs.sync_dir(src_dir_uri, dest_dir_uri, delete=delete)

def start_sync(src_dir_uri, dest_dir_uri, sync_interval=600):
    """Start syncing a directory on a schedule.

    Calls sync_dir on a schedule.

    Args:
        src_dir_uri: (string) URI of source directory
        dest_dir_uri: (string) URI of destination directory
        sync_interval: (int) period in seconds for syncing
    """

    def _sync_dir(delete=True):
        sync_dir(src_dir_uri, dest_dir_uri, delete=delete)
        thread = Timer(sync_interval, _sync_dir)
        thread.daemon = True
        thread.start()

    if urlparse(dest_dir_uri).scheme == 's3':
        # On first sync, we don't want to delete files on S3 to match
        # the contents of output_dir since there's nothing there yet.
        _sync_dir(delete=False)


def download_if_needed(uri, download_dir):
    """Download a file into a directory if it's remote.

    If uri is local, there is no need to download the file.

    Args:
        uri: (string) URI of file
        download_dir: (string) local directory to download file into

    Returns:
        (string) path to local file

    Raises:
        NotReadableError if URI cannot be read from
    """
    if uri is None:
        return None

    path = get_local_path(uri, download_dir)
    make_dir(path, use_dirname=True)

    print('Downloading {} to {}'.format(uri, path))

    fs = FileSystem.get_file_system(uri)
    fs.copy_from(uri, path)

    return path


def download_or_copy(uri, target_dir):
    """Downloads or copies a file to a directory

    Args:
       uri: (string) URI of file
       target_dir: (string) local directory to copy file to
    """
    local_path = download_if_needed(uri, target_dir)
    shutil.copy(local_path, target_dir)
    return local_path


def file_exists(uri):
    fs = FileSystem.get_file_system(uri)
    return fs.file_exists(uri)


def upload_or_copy(src_path, dst_uri):
    """Upload a file if the destination is remote.

    If dst_uri is local, the file is copied.

    Args:
        src_path: (string) path to source file
        dst_uri: (string) URI of destination for file

    Raises:
        NotWritableError if URI cannot be written to
    """
    if dst_uri is None:
        return

    if not (os.path.isfile(src_path) or os.path.isdir(src_path)):
        raise Exception('{} does not exist.'.format(src_path))

    print('Uploading {} to {}'.format(src_path, dst_uri))

    fs = FileSystem.get_file_system(dst_uri)
    fs.copy_to(src_path, dst_uri)

def file_to_str(uri):
    """Download contents of text file into a string.

    Args:
        uri: (string) URI of file

    Returns:
        (string) with contents of text file

    Raises:
        NotReadableError if URI cannot be read from
    """
    fs = FileSystem.get_file_system(uri)
    return fs.read_str(uri)


def str_to_file(content_str, uri):
    """Writes string to text file.

    Args:
        content_str: string to write
        uri: (string) URI of file to write

    Raise:
        NotWritableError if file_uri cannot be written
    """
    fs = FileSystem.get_file_system(uri)
    return fs.write_str(uri, content_str)


def load_json_config(uri, message):
    """Load a JSON-formatted protobuf config file.

    Args:
        uri: (string) URI of config file
        message: (google.protobuf.message.Message) empty protobuf message of
            to load the config into. The type needs to match the content of
            uri.

    Returns:
        the same message passed as input with fields filled in from uri

    Raises:
        ProtobufParseException if uri cannot be parsed
    """
    try:
        return json_format.Parse(file_to_str(uri), message)
    except json_format.ParseError as e:
        error_msg = ('Problem parsing protobuf file {}. '.format(uri) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg) from e


def save_json_config(message, uri):
    """Save a protobuf object to a JSON file.

    Args:
        message: (google.protobuf.message.Message) protobuf message
        uri: (string) URI of JSON file to write message to

    Raises:
        NotWritableError if uri cannot be written
    """
    json_str = json_format.MessageToJson(message)
    str_to_file(json_str, uri)


# Ensure that RV temp directory exists. We need to use a custom location for
# the temporary directory so it will be mirrored on the host file system which
# is needed for running in a Docker container with limited space on EC2.
RV_TEMP_DIR = '/opt/data/tmp/'

# find explicitly set tempdir
explicit_temp_dir = next(
    iter([
        os.environ.get(k) for k in ['TMPDIR', 'TEMP', 'TMP'] if k in os.environ
    ] + [tempfile.tempdir]))

try:
    # try to create directory
    if not os.path.exists(explicit_temp_dir):
        os.makedirs(explicit_temp_dir, exist_ok=True)
    # can we interact with directory?
    explicit_temp_dir_valid = (os.path.isdir(explicit_temp_dir) and Path.touch(
        Path(os.path.join(explicit_temp_dir, '.can_touch'))))
except Exception:
    print('Root temporary directory cannot be used: {}. Using root: {}'.format(
        explicit_temp_dir, RV_TEMP_DIR))
    tempfile.tempdir = RV_TEMP_DIR  # no guarantee this will work
    make_dir(RV_TEMP_DIR)
finally:
    # now, ensure uniqueness for this process
    # the host may be running more than one rastervision process
    RV_TEMP_DIR = tempfile.mkdtemp()
    tempfile.tempdir = RV_TEMP_DIR
    print('Temporary directory is: {}'.format(tempfile.tempdir))
