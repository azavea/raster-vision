import io
import os

import boto3
import botocore
import numpy as np
import shutil
import subprocess
import tempfile

from google.protobuf import json_format
from pathlib import Path
from PIL import Image, ImageColor
from threading import Timer
from urllib.parse import urlparse


class NotReadableError(Exception):
    pass


class NotWritableError(Exception):
    pass


class ProtobufParseException(Exception):
    pass


def color_to_integer(color: str) -> int:
    """Given a PIL ImageColor string, return a packed integer.

    Args:
         color: A PIL ImageColor string

    Returns:
         An integer containing the packed RGB values.

    """
    try:
        triple = ImageColor.getrgb(color)
    except ValueError:
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        triple = (r, g, b)

    r = triple[0] * (1 << 16)
    g = triple[1] * (1 << 8)
    b = triple[2] * (1 << 0)
    integer = r + g + b
    return integer


def numpy_to_png(array: np.ndarray) -> str:
    """Get a PNG string from a Numpy array.

    Args:
         array: A Numpy array of shape (w, h, 3) or (w, h), where the
               former is meant to become a three-channel image and the
               latter a one-channel image.  The dtype of the array
               should be uint8.

    Returns:
         str

    """
    im = Image.fromarray(array)
    output = io.BytesIO()
    im.save(output, 'png')
    return output.getvalue()


def png_to_numpy(png: str, dtype=np.uint8) -> np.ndarray:
    """Get a Numpy array from a PNG string.

    Args:
         png: A str containing a PNG-formatted image.

    Returns:
         numpy.ndarray

    """
    incoming = io.BytesIO(png)
    im = Image.open(incoming)
    return np.array(im)


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
    """Make a directory.

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

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == '':
        path = uri
    elif parsed_uri.scheme == 's3':
        path = os.path.join(download_dir, 's3', parsed_uri.netloc,
                            parsed_uri.path[1:])

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
    command = ['aws', 's3', 'sync', src_dir_uri, dest_dir_uri]
    if delete:
        command.append('--delete')
    subprocess.run(command)


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

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 's3':
        try:
            print('Downloading {} to {}'.format(uri, path))
            s3 = boto3.client('s3')
            s3.download_file(parsed_uri.netloc, parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError:
            raise NotReadableError('Could not read {}'.format(uri))
    else:
        not_found = not os.path.isfile(path)
        if not_found:
            raise NotReadableError('Could not read {}'.format(uri))

    return path


def upload_if_needed(src_path, dst_uri):
    """Upload a file if the destination is remote.

    If dst_uri is local, there is no need to upload.

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

    parsed_uri = urlparse(dst_uri)
    if parsed_uri.scheme == 's3':
        # Strip the leading slash off of the path since S3 does not expect it.
        print('Uploading {} to {}'.format(src_path, dst_uri))
        if os.path.isfile(src_path):
            try:
                s3 = boto3.client('s3')
                s3.upload_file(src_path, parsed_uri.netloc,
                               parsed_uri.path[1:])
            except Exception:
                raise NotWritableError('Could not write {}'.format(dst_uri))
        else:
            sync_dir(src_path, dst_uri, delete=True)


def file_to_str(file_uri):
    """Download contents of text file into a string.

    Args:
        file_uri: (string) URI of file

    Returns:
        (string) with contents of text file

    Raises:
        NotReadableError if URI cannot be read from
    """

    parsed_uri = urlparse(file_uri)
    if parsed_uri.scheme == 's3':
        with io.BytesIO() as file_buffer:
            try:
                s3 = boto3.client('s3')
                s3.download_fileobj(parsed_uri.netloc, parsed_uri.path[1:],
                                    file_buffer)
                return file_buffer.getvalue().decode('utf-8')
            except botocore.exceptions.ClientError:
                raise NotReadableError('Could not read {}'.format(file_uri))
    else:
        if not os.path.isfile(file_uri):
            raise NotReadableError('Could not read {}'.format(file_uri))
        with open(file_uri, 'r') as file_buffer:
            return file_buffer.read()


def str_to_file(content_str, file_uri):
    """Writes string to text file.

    Args:
        content_str: string to write
        file_uri: (string) URI of file to write

    Raise:
        NotWritableError if file_uri cannot be written
    """
    parsed_uri = urlparse(file_uri)
    if parsed_uri.scheme == 's3':
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        with io.BytesIO(bytes(content_str, encoding='utf-8')) as str_buffer:
            try:
                s3 = boto3.client('s3')
                s3.upload_fileobj(str_buffer, bucket, key)
            except Exception:
                raise NotWritableError('Could not write {}'.format(file_uri))
    else:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'w') as content_file:
            content_file.write(content_str)


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
    except json_format.ParseError:
        error_msg = ('Problem parsing protobuf file {}. '.format(uri) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg)


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
