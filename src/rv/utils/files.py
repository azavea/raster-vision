import os
import shutil
from urllib.parse import urlparse
import subprocess

import boto3
import botocore

s3 = boto3.resource('s3')


class NotFoundException(Exception):
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
        path = os.path.join(
            temp_dir, 's3', parsed_uri.netloc, parsed_uri.path[1:])

    return path


def sync_dir(src_dir, dest_uri, delete=False):
    command = ['aws', 's3', 'sync', src_dir, dest_uri]
    if delete:
        command.append('--delete')
    subprocess.run(command)


def download_if_needed(uri, download_dir, must_exist=True):
    """Download a file into a directory if it's remote."""
    if uri is None:
        return None

    path = get_local_path(uri, download_dir)
    make_dir(path, use_dirname=True)

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme == 's3':
        try:
            print('Downloading {} to {}'.format(uri, path))
            s3.Bucket(parsed_uri.netloc).download_file(
                parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError:
            if must_exist:
                raise NotFoundException('Could not find {}'.format(uri))
    else:
        not_found = not os.path.isfile(path)
        if not_found and must_exist:
            raise NotFoundException('Could not find {}'.format(uri))

    return path


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
            s3.meta.client.upload_file(
                src_path, parsed_uri.netloc, parsed_uri.path[1:])
        else:
            sync_dir(src_path, dst_uri, delete=True)
