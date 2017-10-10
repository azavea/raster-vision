from os.path import join, isdir, isfile, dirname
from shutil import rmtree
from os import makedirs
from urllib.parse import urlparse
from subprocess import run
import signal
from ctypes import cdll
from time import sleep
import json
import pandas

import numpy as np
import boto3
import botocore

from object_detection.utils import np_box_list

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
    """Load a window of an image from a TIFF file.

    Args:
        window: ((row_start, row_stop), (col_start, col_stop)) or
        ((y_min, y_max), (x_min, x_max))
    """
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
        # String the leading slash off of the path since S3 does not expect it.
        s3.meta.client.upload_file(
            src_path, parsed_uri.netloc, parsed_uri.path[1:])


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
    """Extract boxes and related info from GeoJSON file

    Returns boxes, classes, scores, where each is a numpy array. The
    array of boxes has shape [N, 4], where the columns correspond to
    ymin, xmin, ymax, and xmax.
    """
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

        box = (ymin, xmin, ymax, xmax)
        boxes.append(box)

        # Get class_id if exists, else use default of 1.
        if 'properties' in feature:
            if 'class_id' in feature['properties']:
                class_id = feature['properties']['class_id']
                box_to_class_id[box] = class_id
            if 'score' in feature['properties']:
                score = feature['properties']['score']
                box_to_score[box] = score

    # Remove duplicates. Needed for ships dataset.
    boxes = list(set(boxes))
    classes = np.array([box_to_class_id.get(box, 1) for box in boxes],
                       dtype=int)
    scores = np.array([box_to_score.get(box) for box in boxes], dtype=float)
    boxes = np.array(boxes, dtype=float)

    return boxes, classes, scores


def translate_boxlist(boxlist, x_offset, y_offset):
    """Translate box coordinates by an offset.

    Args:
    boxlist: BoxList holding N boxes
    x_offset: float
    y_offset: float

    Returns:
    boxlist: BoxList holding N boxes
    """
    y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
    y_min = y_min + y_offset
    y_max = y_max + y_offset
    x_min = x_min + x_offset
    x_max = x_max + x_offset
    translated_boxlist = np_box_list.BoxList(
       np.hstack([y_min, x_min, y_max, x_max]))

    fields = boxlist.get_extra_fields()
    for field in fields:
        extra_field_data = boxlist.get_field(field)
        translated_boxlist.add_field(field, extra_field_data)

    return translated_boxlist
