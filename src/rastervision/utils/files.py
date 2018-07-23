import io
import os
from pathlib import Path
import rasterio
import shutil
import subprocess
import tempfile
from threading import Timer
from urllib.parse import urlparse

import boto3
import botocore
from google.protobuf import json_format


class NotFoundException(Exception):
    pass


class NotWritableError(Exception):
    pass


class ProtobufParseException(Exception):
    pass


def make_dir(path, check_empty=False, force_empty=False, use_dirname=False):
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


def get_local_path(uri, temp_dir):
    """Convert a URI into a corresponding local path."""
    if uri is None:
        return None

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == '':
        path = uri
    elif parsed_uri.scheme == 's3':
        path = os.path.join(temp_dir, 's3', parsed_uri.netloc,
                            parsed_uri.path[1:])

    return path


def sync_dir(src_dir, dest_uri, delete=False):
    command = ['aws', 's3', 'sync', src_dir, dest_uri]
    if delete:
        command.append('--delete')
    subprocess.run(command)


def _is_raster(uri, s3_test=False):
    if s3_test:
        uri = uri.replace('s3://', '/vsis3/')

    try:
        rasterio.open(uri)
    except Exception:
        return False
    return uri


def download_if_needed(uri, download_dir):
    """Download a file into a directory if it's remote."""
    if uri is None:
        return None

    path = get_local_path(uri, download_dir)
    make_dir(path, use_dirname=True)

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 's3':
        vsis3_path = _is_raster(uri, s3_test=True)
        if vsis3_path:
            return vsis3_path

        try:
            print('Downloading {} to {}'.format(uri, path))
            s3 = boto3.client('s3')
            s3.download_file(parsed_uri.netloc, parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError:
            raise NotFoundException('Could not access {}'.format(uri))
    else:
        not_found = not os.path.isfile(path)
        if not_found:
            raise NotFoundException('Could not access {}'.format(uri))

    return path


def file_to_str(file_uri):
    parsed_uri = urlparse(file_uri)
    if parsed_uri.scheme == 's3':
        with io.BytesIO() as file_buffer:
            try:
                s3 = boto3.client('s3')
                s3.download_fileobj(parsed_uri.netloc, parsed_uri.path[1:],
                                    file_buffer)
                return file_buffer.getvalue().decode('utf-8')
            except botocore.exceptions.ClientError:
                raise NotFoundException('Could not access {}'.format(file_uri))
    else:
        if not os.path.isfile(file_uri):
            raise NotFoundException('Could not access {}'.format(file_uri))
        with open(file_uri, 'r') as file_buffer:
            return file_buffer.read()


def str_to_file(content_str, file_uri):
    parsed_uri = urlparse(file_uri)
    if parsed_uri.scheme == 's3':
        bucket = parsed_uri.netloc
        key = parsed_uri.path[1:]
        with io.BytesIO(bytes(content_str, encoding='utf-8')) as str_buffer:
            s3 = boto3.client('s3')
            s3.upload_fileobj(str_buffer, bucket, key)
    else:
        make_dir(file_uri, use_dirname=True)
        with open(file_uri, 'w') as content_file:
            content_file.write(content_str)


def load_json_config(uri, message):
    try:
        return json_format.Parse(file_to_str(uri), message)
    except json_format.ParseError:
        error_msg = ('Problem parsing protobuf file {}. '.format(uri) +
                     'You might need to run scripts/compile')
        raise ProtobufParseException(error_msg)


def save_json_config(message, uri):
    json_str = json_format.MessageToJson(message)
    str_to_file(json_str, uri)


def upload_if_needed(src_path, dst_uri):
    """Upload file or dir if the destination is remote."""
    if dst_uri is None:
        return

    if not (os.path.isfile(src_path) or os.path.isdir(src_path)):
        raise Exception('{} does not exist.'.format(src_path))

    parsed_uri = urlparse(dst_uri)
    if parsed_uri.scheme == 's3':
        # Strip the leading slash off of the path since S3 does not expect it.
        print('Uploading {} to {}'.format(src_path, dst_uri))
        if os.path.isfile(src_path):
            s3 = boto3.client('s3')
            s3.upload_file(src_path, parsed_uri.netloc, parsed_uri.path[1:])
        else:
            sync_dir(src_path, dst_uri, delete=True)


def start_sync(output_dir, output_uri, sync_interval=600):
    """Start periodically syncing a directory."""

    def _sync_dir(delete=True):
        sync_dir(output_dir, output_uri, delete=delete)
        thread = Timer(sync_interval, _sync_dir)
        thread.daemon = True
        thread.start()

    if urlparse(output_uri).scheme == 's3':
        # On first sync, we don't want to delete files on S3 to match
        # th contents of output_dir since there's nothing there yet.
        _sync_dir(delete=False)


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
