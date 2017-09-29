from os.path import join, isdir, isfile, dirname
from shutil import rmtree
from os import makedirs
from urllib.parse import urlparse
from subprocess import run
import signal
from ctypes import cdll
from time import sleep
import json

import numpy as np
import boto3
import botocore

s3 = boto3.resource('s3')

# Constant taken from http://linux.die.net/include/linux/prctl.h
PR_SET_PDEATHSIG = 1


class PrCtlError(Exception):
    pass


# From http://evans.io/legacy/posts/killing-child-processes-on-parent-exit-prctl/  # noqa
def on_parent_exit(signame):
    """
    Return a function to be run in a child process which will trigger
    SIGNAME to be sent when the parent process dies
    """
    signum = getattr(signal, signame)

    def set_parent_exit_signal():
        # http://linux.die.net/man/2/prctl
        result = cdll['libc.so.6'].prctl(PR_SET_PDEATHSIG, signum)
        if result != 0:
            raise PrCtlError('prctl failed with error code %s' % result)
    return set_parent_exit_signal


def load_window(image_dataset, channel_order, window=None):
    im = np.transpose(
        image_dataset.read(window=window), axes=[1, 2, 0])
    im = im[:, :, channel_order]
    return im


def get_local_path(temp_dir, uri):
    """Convert a URI into a corresponding local path."""
    if uri is None:
        return None

    parsed_uri = urlparse(uri)
    path = parsed_uri.path[1:] if parsed_uri.path[0] == '/' \
        else parsed_uri.path
    if parsed_uri.scheme == 'file':
        path = join(parsed_uri.netloc, path)
    elif parsed_uri.scheme == '':
        path = uri
    elif parsed_uri.scheme == 's3':
        path = join(temp_dir, path)

    return path


class NotFoundException(Exception):
    pass


def download_if_needed(download_dir, uri, must_exist=True):
    """Download a file into a directory if it's remote."""
    if uri is None:
        return None

    path = get_local_path(download_dir, uri)
    parsed_uri = urlparse(uri)
    not_found = False
    if parsed_uri.scheme == 's3':
        makedirs(dirname(path), exist_ok=True)
        try:
            s3.Bucket(parsed_uri.netloc).download_file(
                parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                not_found = True
    else:
        not_found = not isfile(path)

    if not_found and must_exist:
        raise NotFoundException('{} does not exist.'.format(uri))

    return path


def upload_if_needed(src_path, dst_uri):
    """Upload file if the destination is remote."""
    if dst_uri is None:
        return

    if not isfile(src_path):
        raise Exception('{} does not exist.'.format(src_path))

    parsed_uri = urlparse(dst_uri)
    if parsed_uri.scheme == 's3':
        s3.meta.client.upload_file(
            src_path, parsed_uri.netloc, parsed_uri.path)


def build_vrt(vrt_path, image_paths):
    """Build a VRT for a set of TIFF files."""
    cmd = ['gdalbuildvrt', vrt_path]
    cmd.extend(image_paths)
    run(cmd)


def download_and_build_vrt(temp_dir, image_uris):
    image_paths = [download_if_needed(temp_dir, uri) for uri in image_uris]
    image_path = join(temp_dir, 'index.vrt')
    build_vrt(image_path, image_paths)
    return image_path


def make_temp_dir(temp_dir):
    if isdir(temp_dir):
        rmtree(temp_dir)
    makedirs(temp_dir, exist_ok=True)


def sync_dir(src_dir, dest_uri):
    run(['aws', 's3', 'sync', src_dir, dest_uri, '--delete'])


def get_boxes_from_geojson(json_path, image_dataset):
    with open(json_path, 'r') as json_file:
        geojson = json.load(json_file)

    features = geojson['features']
    boxes = []
    box_to_class_id = {}
    box_to_score = {}

    for feature in features:
        polygon = feature['geometry']['coordinates'][0]
        # Convert to pixel coords.
        polygon = [image_dataset.index(p[0], p[1]) for p in polygon]
        polygon = np.array([(p[1], p[0]) for p in polygon])

        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        box = (xmin, ymin, xmax, ymax)
        boxes.append(box)

        # Get class_id if exists, else use default of 1.
        class_id = 1
        score = None
        if 'properties' in feature:
            if 'class_id' in feature['properties']:
                class_id = feature['properties']['class_id']
            if 'score' in feature['properties']:
                score = feature['properties']['score']
        box_to_class_id[box] = class_id
        box_to_score[box] = score

    # Remove duplicates. Needed for ships dataset.
    boxes = list(set(boxes))
    return boxes, box_to_class_id, box_to_score
