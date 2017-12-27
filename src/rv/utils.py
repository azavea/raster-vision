from os.path import join, isdir, isfile, dirname
from shutil import rmtree
from os import makedirs
from urllib.parse import urlparse
from subprocess import run
import signal
from ctypes import cdll
import json

from scipy.misc import imsave
from pyproj import Proj, transform
import numpy as np
import boto3
import botocore
from rtree import index

from object_detection.utils import np_box_list

s3 = boto3.resource('s3')

# Constant taken from http://linux.die.net/include/linux/prctl.h
PR_SET_PDEATHSIG = 1


class PrCtlError(Exception):
    pass


def load_projects(temp_dir, projects_path):
    image_paths_list = []
    annotations_paths = []
    with open(projects_path, 'r') as projects_file:
        projects = json.load(projects_file)
        for project in projects:
            image_uris = project['images']
            image_paths = [download_if_needed(temp_dir, image_uri)
                           for image_uri in image_uris]
            image_paths_list.append(image_paths)
            annotations_uri = project['annotations']
            annotations_path = download_if_needed(temp_dir, annotations_uri)
            annotations_paths.append(annotations_path)

    return image_paths_list, annotations_paths


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
    if parsed_uri.scheme == 's3':
        makedirs(dirname(path), exist_ok=True)
        try:
            print('Downloading {} to {}'.format(uri, path))
            s3.Bucket(parsed_uri.netloc).download_file(
                parsed_uri.path[1:], path)
        except botocore.exceptions.ClientError as e:
            if must_exist:
                raise e
    else:
        not_found = not isfile(path)
        if not_found and must_exist:
            raise NotFoundException('Could not find {}'.format(uri))

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
        print('Uploading {} to {}'.format(src_path, dst_uri))
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


def make_empty_dir(temp_dir, empty_dir=True):
    if empty_dir:
        if isdir(temp_dir):
            rmtree(temp_dir)
    makedirs(temp_dir, exist_ok=True)


def sync_dir(src_dir, dest_uri, delete=False):
    command = ['aws', 's3', 'sync', src_dir, dest_uri]
    if delete:
        command.append('--delete')
    run(command)


def get_boxes_from_geojson(json_path, image_dataset, label_map=None):
    """Extract boxes and related info from GeoJSON file

    Returns boxes, classes, scores, where each is a numpy array. The
    array of boxes has shape [N, 4], where the columns correspond to
    ymin, xmin, ymax, and xmax.
    """
    with open(json_path, 'r') as json_file:
        geojson = json.load(json_file)

    # Convert from lat/lng to image_dataset CRS
    src_crs = 'epsg:4326'
    src_proj = Proj(init=src_crs)
    dst_crs = image_dataset.crs['init']
    dst_proj = Proj(init=dst_crs)

    features = geojson['features']
    boxes = []
    box_to_class_id = {}
    box_to_score = {}

    for feature in features:
        polygon = feature['geometry']['coordinates'][0]
        # Convert to image_dataset CRS and then pixel coords.
        polygon = [transform(src_proj, dst_proj, p[0], p[1]) for p in polygon]
        polygon = [image_dataset.index(p[0], p[1]) for p in polygon]
        polygon = np.array([(p[1], p[0]) for p in polygon])

        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)

        box = (ymin, xmin, ymax, xmax)
        boxes.append(box)

        if 'properties' in feature:
            class_id = 1
            if 'class_id' in feature['properties']:
                class_id = feature['properties']['class_id']
            elif 'label' in feature['properties'] and label_map is not None:
                class_id = label_map[feature['properties']['label']]
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


def save_geojson(path, boxlist, category_index=None, image_dataset=None):
    if image_dataset:
        src_crs = image_dataset.crs['init']
        src_proj = Proj(init=src_crs)
        # Convert to lat/lng
        dst_crs = 'epsg:4326'
        dst_proj = Proj(init=dst_crs)

    polygons = []
    for box in boxlist.get():
        ymin, xmin, ymax, xmax = box

        # four corners
        nw = (ymin, xmin)
        ne = (ymin, xmax)
        se = (ymax, xmax)
        sw = (ymax, xmin)
        polygon = [nw, ne, se, sw, nw]
        # Transform from pixel coords to spatial coords
        if image_dataset:
            dst_polygon = []
            for point in polygon:
                src_crs_point = image_dataset.ul(point[0], point[1])
                dst_crs_point = transform(
                    src_proj, dst_proj, src_crs_point[0], src_crs_point[1])
                dst_polygon.append(dst_crs_point)
        polygons.append(dst_polygon)

    crs = None
    if image_dataset:
        crs = {
            'type': 'name',
            'properties': {
                'name': dst_crs
            }
        }

    features = []
    for ind, polygon in enumerate(polygons):
        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [polygon]
            }
        }

        if boxlist.has_field('classes') and boxlist.has_field('scores'):
            classes = boxlist.get_field('classes')
            scores = boxlist.get_field('scores')
            class_id, score = classes[ind], scores[ind]
            feature['properties'] = {
                'class_id': int(class_id),
                'class_name': category_index[class_id]['name'],
                'score': score
            }

        features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'crs': crs,
        'features': features
    }

    with open(path, 'w') as json_file:
        json.dump(geojson, json_file, indent=4)


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


def save_img(path, arr):
    imsave(path, arr)


class BoxDB():
    def __init__(self, boxes):
        """Build DB of boxes for fast intersection queries

        Args:
            boxes: [N, 4] numpy array of boxes with cols ymin, xmin, ymax, xmax
        """
        self.boxes = boxes
        self.rtree_idx = index.Index()
        for box_ind, box in enumerate(boxes):
            # rtree order is xmin, ymin, xmax, ymax
            rtree_box = (box[1], box[0], box[3], box[2])
            self.rtree_idx.insert(box_ind, rtree_box)

    def get_intersecting_box_inds(self, x, y, box_size):
        query_box = (x, y, x + box_size, y + box_size)
        intersection_inds = list(self.rtree_idx.intersection(query_box))
        return intersection_inds


def print_box_stats(boxes):
    print('# boxes: {}'.format(len(boxes)))

    ymins, xmins, ymaxs, xmaxs = boxes.T
    width = xmaxs - xmins + 1
    print('width (mean, min, max): ({}, {}, {})'.format(
        np.mean(width), np.min(width), np.max(width)))

    height = ymaxs - ymins + 1
    print('height (mean, min, max): ({}, {}, {})'.format(
        np.mean(height), np.min(height), np.max(height)))


def add_blank_chips(blank_count, chip_size, chip_dir):
    blank_im = np.zeros((chip_size, chip_size, 3))
    for blank_neg_ind in range(blank_count):
        chip_path = join(chip_dir, 'blank-{}.png'.format(blank_neg_ind))
        save_img(chip_path, blank_im)


def get_random_window_for_box(box, im_width, im_height, chip_size):
    """Get random window in image that contains box.

    Returns: upper-left corner of window
    """
    ymin, xmin, ymax, xmax = box

    # ensure that window doesn't go off the edge of the array.
    width = xmax - xmin
    lb = max(0, xmin - (chip_size - width))
    ub = min(im_width - chip_size, xmin)
    rand_x = int(np.random.uniform(lb, ub))

    height = ymax - ymin
    lb = max(0, ymin - (chip_size - height))
    ub = min(im_height - chip_size, ymin)
    rand_y = int(np.random.uniform(lb, ub))

    return (rand_x, rand_y)


def get_random_window(im_width, im_height, chip_size):
    """Get random window somewhere in image.

    Returns: upper-left corner of window
    """
    rand_x = int(np.random.uniform(0, im_width - chip_size))
    rand_y = int(np.random.uniform(0, im_height - chip_size))
    return (rand_x, rand_y)
